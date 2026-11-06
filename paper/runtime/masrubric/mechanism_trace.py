"""Record audit attempts and terminal communication decisions without model calls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

_LOCK = threading.Lock()


def code_max_attempts():
    budget = int(os.environ.get("MASRUBRIC_CODE_RETRY_BUDGET", "3"))
    if budget < 0:
        raise ValueError("MASRUBRIC_CODE_RETRY_BUDGET must be nonnegative")
    return budget + 1


class AuditMechanismRecorder:
    def __init__(self):
        self.withholding_policy = os.environ.get("MASRUBRIC_WITHHOLDING_POLICY", "withhold")
        if self.withholding_policy not in ("withhold", "broadcast"):
            raise ValueError("MASRUBRIC_WITHHOLDING_POLICY must be withhold or broadcast")
        self.trace_path = os.environ.get("MASRUBRIC_MECHANISM_TRACE", "")
        self.session_id = str(uuid4())
        self.phase = 0
        self.pending = {}
        self.contexts = {}

    def emit(self, event_type, **payload):
        if not self.trace_path:
            return
        event = dict(schema_version=1, event_type=event_type, session_id=self.session_id,
                     phase=self.phase, timestamp=datetime.now(timezone.utc).isoformat(), **payload)
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def remember_context(self, recipient, messages):
        if recipient:
            self.contexts[recipient] = [{"id": m.id, "source": m.source, "content": m.content}
                                        for m in messages]

    def record(self, *, task, message, attempt, role, metrics, judgments, passed,
               selection, failures, feedback=None, skip_reason=None):
        source = message.source
        if attempt == 1 or source not in self.pending:
            self.pending[source] = []
        if skip_reason:
            empty_cause = skip_reason
        elif judgments:
            empty_cause = None
        elif not metrics:
            empty_cause = "no_selected_criteria"
        elif failures:
            empty_cause = "all_selected_audits_failed"
        else:
            empty_cause = "selected_criteria_without_judgments"
        record = dict(attempt=attempt, question=task, message=message.content, source=source,
                      role=role, audit_message_id=message.id,
                      input_messages=deepcopy(self.contexts.get(source, [])),
                      selected_metrics=[m["name"] for m in metrics], selected_count=len(metrics),
                      judgements=deepcopy(judgments), judgment_count=len(judgments),
                      pass_flag=passed, pass_rate_observed=(sum(j.get("verdict", "").lower() == "correct"
                           for j in judgments) / len(judgments)) if judgments else None,
                      empty_cause=empty_cause, empty_policy="pass", selection=deepcopy(selection),
                      audit_failures=deepcopy(failures), feedback=feedback,
                      withholding_policy=self.withholding_policy,
                      action="pass" if passed else "failed_pending_terminal_decision")
        self.pending[source].append(record)
        self.emit("audit_attempt", **record)

    def finalize(self, message, judgments, audit_enabled, failed):
        attempts = self.pending.pop(message.source, [])
        withheld = bool(audit_enabled and failed and self.withholding_policy == "withhold")
        action = ("withhold" if withheld else "broadcast_after_failed_audit" if audit_enabled and failed
                  else "broadcast" if audit_enabled else "broadcast_audit_disabled")
        for index, attempt in enumerate(attempts):
            attempt["action"] = "revise" if index < len(attempts) - 1 else action
        details = dict(audit_failed=bool(audit_enabled and failed),
                       withholding_policy=self.withholding_policy, empty_policy="pass",
                       action=action, audit_attempts=attempts,
                       mechanism_session_id=self.session_id, mechanism_phase=self.phase)
        self.emit("message_finalized", message_id=message.id, source=message.source,
                  message=message.content, judgements=deepcopy(judgments), is_pruned=withheld, **details)
        return withheld, details

    def reset(self, scoreboard):
        self.emit("scoreboard_reset", discarded_message_ids=list(scoreboard),
                  retained_before_reset=sum(not row["is_pruned"] for row in scoreboard.values()))
        self.phase += 1
        self.pending.clear()
        self.contexts.clear()
