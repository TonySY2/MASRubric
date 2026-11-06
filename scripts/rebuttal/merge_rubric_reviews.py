"""Read two independently completed XLSX reviews; preserve unknowns and adjudication.

Uses only Python's standard library. Never writes an XLSX or modifies inputs.
"""
import argparse
import csv
import datetime as dt
import io
import json
import posixpath
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

DIMENSIONS = ['语义正确', '可执行', '可泛化', '无需测试答案']
HEADERS = ['ID', '领域', 'name', 'definition', 'condition', 'directive'] + DIMENSIONS + ['理由']
VALID = {'是', '否', '不确定'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def read_xlsx(path):
    """Return worksheets as cell-address dictionaries, rejecting formulas."""
    sheets = {}
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        book = ET.fromstring(z.read('xl/workbook.xml'))
        for s in book.findall('s:sheets/s:sheet', NS):
            target = rels[s.attrib['{' + REL_NS + '}id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
            cells = {}
            for c in ET.fromstring(z.read(target)).findall('.//s:sheetData/s:row/s:c', NS):
                if c.find('s:f', NS) is not None:
                    raise ValueError(f"{path}: {s.attrib['name']}!{c.attrib['r']} contains a formula; labels and original text must be literal values")
                value = c.find('s:v', NS)
                if c.attrib.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in c.findall('.//s:t', NS))
                elif c.attrib.get('t') == 's':
                    value = strings[int(value.text)] if value is not None else ''
                else:
                    value = value.text if value is not None and value.text is not None else ''
                cells[c.attrib['r']] = value
            sheets[s.attrib['name']] = cells
    return sheets


def normalize_date(value):
    value = str(value).strip()
    if not value:
        return ''
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError:
        try:
            serial = float(value)
            if 30000 <= serial <= 100000:
                return (dt.datetime(1899, 12, 30) + dt.timedelta(days=serial)).date().isoformat()
        except ValueError:
            pass
    raise ValueError('填写日期须为Excel日期或YYYY-MM-DD')


def parse_review(path, key):
    sheets = read_xlsx(path)
    table = sheets[key['sheet']]
    meta = sheets[key['protocol_sheet']]
    for col, expected in zip('ABCDEFGHIJK', HEADERS):
        if table.get(col + '1', '') != expected:
            raise ValueError(f'{path}: header {col}1 changed')
    expected_rows = {r['id']: r for r in key['rows']}
    result = {}
    rownums = sorted({int(a.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) for a in table if a[0].isalpha()} - {1})
    for n in rownums:
        values = [str(table.get(c + str(n), '')) for c in 'ABCDEFGHIJK']
        if not any(values):
            continue
        blind_id = values[0]
        if blind_id not in expected_rows:
            raise ValueError(f'{path}: unknown/blank ID at row {n}: {blind_id!r}')
        if blind_id in result:
            raise ValueError(f'{path}: duplicate ID {blind_id}')
        source = expected_rows[blind_id]
        fixed = [blind_id, source['domain'], source['name'], source['definition'], source['condition'], source['directive']]
        if values[:6] != fixed:
            raise ValueError(f'{path}: original text or domain changed for {blind_id}')
        labels = [x.strip() for x in values[6:10]]
        invalid = [v for v in labels if v and v not in VALID]
        if invalid:
            raise ValueError(f'{path}: invalid labels {invalid} for {blind_id}')
        result[blind_id] = {'labels': labels, 'reason': values[10].strip()}
    missing = sorted(set(expected_rows) - set(result))
    if missing:
        raise ValueError(f'{path}: missing IDs: {missing[:10]} ({len(missing)} total)')
    name = str(meta.get(key['reviewer_cell'], '')).strip()
    date = normalize_date(meta.get(key['date_cell'], ''))
    declared = str(meta.get(key['completion_cell'], '')).strip()
    if declared not in {'', '是', '否'}:
        raise ValueError(f'{path}: invalid completion confirmation')
    blanks = [(i, DIMENSIONS[j]) for i, r in result.items() for j, v in enumerate(r['labels']) if not v]
    missing_reasons = [i for i, r in result.items() if any(v in {'否', '不确定'} for v in r['labels']) and not r['reason']]
    reasons = []
    if declared != '是': reasons.append('未明确确认完成')
    if not name: reasons.append('未填写标注者姓名')
    if not date: reasons.append('未填写日期')
    if blanks: reasons.append(f'{len(blanks)}个标签未填')
    if missing_reasons: reasons.append(f'{len(missing_reasons)}条缺必要理由')
    return {'source': str(path), 'reviewer': name, 'date': date, 'declared_complete': declared == '是',
            'complete': not reasons, 'completion_issues': reasons, 'blank_labels': blanks,
            'missing_reasons': missing_reasons, 'rows': result}


def label_counts(values, allow_rates):
    c = Counter(values)
    known = c['是'] + c['否']
    return {'yes': c['是'], 'no': c['否'], 'uncertain': c['不确定'], 'blank_or_unresolved': c[''],
            'determinate_denominator': known, 'yes_rate_among_determinate': c['是'] / known if known and allow_rates else None}


def read_adjudication(path, disputes):
    if path is None:
        return {}
    expected = {(d['ID'], d['维度']): d for d in disputes}
    resolutions = {}
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            k = (r.get('ID', ''), r.get('维度', ''))
            if k not in expected or k in seen:
                raise ValueError(f'裁决表含未知/重复项目: {k}')
            seen.add(k)
            if r.get('标注者A') != expected[k]['标注者A'] or r.get('标注者B') != expected[k]['标注者B']:
                raise ValueError(f'裁决表原标签被改动: {k}')
            resolution = r.get('裁决标签', '').strip()
            if not resolution:
                continue
            if resolution not in VALID:
                raise ValueError(f'裁决标签无效: {k}')
            if not r.get('裁决者', '').strip() or not r.get('裁决理由', '').strip() or r.get('裁决完成', '').strip() != '是':
                raise ValueError(f'裁决填写不完整: {k}')
            if not normalize_date(r.get('裁决日期', '')):
                raise ValueError(f'裁决日期缺失: {k}')
            resolutions[k] = dict(r, 裁决标签=resolution)
    return resolutions


def summarize(a, b, key, adjudication=None):
    complete = a['complete'] and b['complete'] and a['reviewer'].casefold() != b['reviewer'].casefold()
    status = {'both_reviews_complete': complete, 'same_reviewer_name': bool(a['reviewer']) and a['reviewer'].casefold() == b['reviewer'].casefold()}
    disagreements = []
    for item in key['rows']:
        i = item['id']
        for j, dim in enumerate(DIMENSIONS):
            va, vb = a['rows'][i]['labels'][j], b['rows'][i]['labels'][j]
            if va and vb and va != vb:
                disagreements.append({'ID': i, '领域': item['domain'], '维度': dim, '标注者A': va, '标注者B': vb,
                                      'A理由': a['rows'][i]['reason'], 'B理由': b['rows'][i]['reason'], '裁决标签': '',
                                      '裁决理由': '', '裁决者': '', '裁决日期': '', '裁决完成': ''})
    if adjudication and not complete:
        raise ValueError('两份独立审核未完成，不能进入最终裁决')
    resolved = read_adjudication(adjudication, disagreements)
    final = {}
    for item in key['rows']:
        i = item['id']; vals = []
        for j, dim in enumerate(DIMENSIONS):
            va, vb = a['rows'][i]['labels'][j], b['rows'][i]['labels'][j]
            vals.append(va if va == vb and va else resolved.get((i, dim), {}).get('裁决标签', ''))
        final[i] = vals
    unresolved = [(i, DIMENSIONS[j]) for i, vals in final.items() for j, v in enumerate(vals) if not v]
    status.update({'disagreements': len(disagreements), 'adjudicated_disagreements': len(resolved),
                   'unresolved_label_pairs': len(unresolved), 'finalized': complete and not unresolved})
    groups = {}
    for domain in ['数学', '代码']:
        ids = [r['id'] for r in key['rows'] if r['domain'] == domain]
        criteria = {}
        for j, dim in enumerate(DIMENSIONS):
            av = [a['rows'][i]['labels'][j] for i in ids]; bv = [b['rows'][i]['labels'][j] for i in ids]
            pairs = [(x, y) for x, y in zip(av, bv) if x and y]
            determinate = [(x, y) for x, y in pairs if x != '不确定' and y != '不确定']
            criteria[dim] = {'A': label_counts(av, complete), 'B': label_counts(bv, complete),
                            'agreement': {'paired_denominator': len(pairs), 'agree_count': sum(x == y for x, y in pairs),
                                          'rate': sum(x == y for x, y in pairs) / len(pairs) if pairs and complete else None,
                                          'determinate_paired_denominator': len(determinate),
                                          'determinate_rate': sum(x == y for x, y in determinate) / len(determinate) if determinate and complete else None},
                            'final': label_counts([final[i][j] for i in ids], status['finalized'])}
        qualified = sum(all(v == '是' for v in final[i]) for i in ids)
        known = sum(all(v in {'是', '否'} for v in final[i]) for i in ids)
        groups[domain] = {'sample_n': len(ids), 'dimensions': criteria,
                          'overall_reliability': {'all_four_yes': qualified, 'all_four_determinate_denominator': known,
                                                  'uncertain_or_unresolved': len(ids) - known,
                                                  'rate': qualified / known if known and status['finalized'] else None}}
    return {'status': status, 'review_A': {k:v for k,v in a.items() if k != 'rows'},
            'review_B': {k:v for k,v in b.items() if k != 'rows'}, 'domains': groups,
            'unresolved': unresolved, 'final_labels': final,
            'adjudication_records': list(resolved.values())}, disagreements


def csv_text(rows, headers):
    buf = io.StringIO(newline=''); w = csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    for row in rows:
        w.writerow({k: "'" + str(row.get(k, '')) if str(row.get(k, '')).startswith(('=', '+', '-', '@')) else row.get(k, '') for k in headers})
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-a', required=True, type=Path)
    p.add_argument('--review-b', required=True, type=Path)
    p.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] / 'docs' / 'rebuttal' / 'annotation' / 'rubric_review_reference.json', help='Public original-text reference used to validate IDs and unchanged source fields')
    p.add_argument('--adjudication', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.review_a.resolve() == args.review_b.resolve():
        p.error('必须传入两位标注者各自的文件')
    key = json.loads(args.reference.read_text(encoding='utf-8'))
    if len(key['rows']) != 200 or Counter(r['domain'] for r in key['rows']) != {'数学':100, '代码':100} or len({r['id'] for r in key['rows']}) != 200:
        p.error('参考表必须包含200个唯一ID，各领域100条')
    a = parse_review(args.review_a, key); b = parse_review(args.review_b, key)
    report, disputes = summarize(a, b, key, args.adjudication)
    outputs = ['review_statistics.json', 'adjudication_required.csv', 'merged_reviews.csv', 'STATISTICS.md']
    if any((args.out / n).exists() for n in outputs):
        p.error('输出文件已存在；请为本次合并选择新的--out目录')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / outputs[0]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    adjud_headers = ['ID','领域','维度','标注者A','标注者B','A理由','B理由','裁决标签','裁决理由','裁决者','裁决日期','裁决完成']
    remaining = [d for d in disputes if not report['final_labels'][d['ID']][DIMENSIONS.index(d['维度'])]]
    (args.out / outputs[1]).write_text(csv_text(remaining, adjud_headers), encoding='utf-8-sig')
    merged = []
    for row in key['rows']:
        i=row['id']; entry={'ID':i,'领域':row['domain'],'A理由':a['rows'][i]['reason'],'B理由':b['rows'][i]['reason']}
        for j, dim in enumerate(DIMENSIONS):
            entry.update({dim+'_A':a['rows'][i]['labels'][j], dim+'_B':b['rows'][i]['labels'][j], dim+'_final':report['final_labels'][i][j]})
        merged.append(entry)
    (args.out / outputs[2]).write_text(csv_text(merged, list(merged[0])), encoding='utf-8-sig')
    status=report['status']; lines=['# 人工准则复核统计','', f"两份独立审核完成：{status['both_reviews_complete']}。最终裁决完成：{status['finalized']}。", '',
                                  f"分歧标签数：{status['disagreements']}；已裁决：{status['adjudicated_disagreements']}；未解决或未填：{status['unresolved_label_pairs']}。", '',
                                  '未完成时统计率为null；空白不作否，不确定不进入确定项通过率分母。详细分维度数量及分母见review_statistics.json。','',
                                  '| 领域 | 样本数 | 四维均是 | 四维确定分母 | 不确定或未解决 | 最终可靠率 |','| --- | ---: | ---: | ---: | ---: | --- |']
    for domain, g in report['domains'].items():
        v=g['overall_reliability']; rate='待完成' if v['rate'] is None else f"{v['rate']:.2%}"
        lines.append(f"| {domain} | {g['sample_n']} | {v['all_four_yes']} | {v['all_four_determinate_denominator']} | {v['uncertain_or_unresolved']} | {rate} |")
    (args.out / outputs[3]).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
