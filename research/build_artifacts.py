from pathlib import Path
import re
import html
import json
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
md = (ROOT / 'AVGO_2026-09-09.md').read_text()

def inline(s):
    s = html.escape(s)
    s = re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    return s

blocks = []
lines = md.splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if not line.strip():
        i += 1
        continue
    if line.startswith('# '):
        blocks.append('<h1>' + inline(line[2:]) + '</h1>')
    elif line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                rows.append(cells)
            i += 1
        table = '<div class="table-wrap"><table><thead><tr>'
        table += ''.join('<th>' + inline(c) + '</th>' for c in rows[0])
        table += '</tr></thead><tbody>'
        for row in rows[1:]:
            table += '<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in row) + '</tr>'
        blocks.append(table + '</tbody></table></div>')
        continue
    else:
        cls = 'source' if re.match(r'^\d+\. ', line) else ''
        blocks.append('<p class="' + cls + '">' + inline(line) + '</p>')
    i += 1

doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>博通 AVGO 估值与入场研究</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#202124;font:17px/1.9 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:960px;margin:56px auto;padding:0 32px 70px}h1{font-size:32px;line-height:1.35;font-weight:650;margin:0 0 30px}p{margin:0 0 21px}strong{font-weight:650}a{color:#244962;text-underline-offset:3px;overflow-wrap:anywhere}.table-wrap{overflow-x:auto;margin:22px 0 28px}table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.6}th,td{padding:11px 12px;border-bottom:1px solid #d9dce0;text-align:left;vertical-align:top}th{background:#f0f1f2;font-weight:650}tbody tr:nth-child(even){background:#fafafa}.source{font-size:13px;line-height:1.7;color:#50545a;margin-bottom:12px}@media(max-width:600px){main{margin:28px auto;padding:0 18px 45px}body{font-size:16px}h1{font-size:27px}table{font-size:13px}th,td{padding:8px}}@page{size:A4;margin:18mm 17mm}@media print{body{font-size:10.5pt;line-height:1.7}main{max-width:none;margin:0;padding:0}h1{font-size:23pt}p{margin-bottom:11pt;orphans:3;widows:3}table{font-size:9pt}th,td{padding:6pt}tr{break-inside:avoid}.table-wrap{overflow:visible}a{color:#202124}.source{font-size:8.5pt}}
</style><main>''' + '\n'.join(blocks) + '</main></html>'
(ROOT / 'AVGO_2026-09-09.html').write_text(doc)

# Minimal standards-based XLSX with editable formulas and cached numeric results.
def F(formula, value):
    return (formula, value)

price = 368.56
shares = 4.76
owner = (13.665 - 2.019) * 4 / shares
inputs = [
    ['参数', '数值', '单位 / 属性', '来源 / 说明'],
    ['股价', price, 'USD；2026-09-08 regular close', 'S1'],
    ['当前股数', shares, '十亿股；近似数', 'S2'],
    ['Q3 报告自由现金流', 13.665, '十亿美元；事实', 'S3'],
    ['Q3 股权薪酬', 2.019, '十亿美元；事实', 'S3'],
    ['要求回报率', .12, '独立模型假设', '分析选择'],
    ['永续增长率', .04, '独立模型假设', '分析选择'],
    ['TTM 报告 FCF', 39.403, '十亿美元；四季度相加', 'S4'],
    ['FY2026 调整后 EPS', 11.64, 'USD；共识截至 2026-09-07', 'S5'],
    ['Q3 调整后 EPS', 3.32, 'USD；事实', 'S3'],
    ['市值', F('B2*B3', price*shares), '十亿美元', '计算'],
    ['TTM FCF 倍数', F('B11/B8', price*shares/39.403), '倍', '计算'],
    ['TTM FCF 收益率', F('B8/B11',39.403/(price*shares)), '比率', '计算'],
    ['FY2026 调整后 PE', F('B2/B9',price/11.64), '倍', '计算'],
    ['Q3 EPS 年化 PE', F('B2/(B10*4)',price/(3.32*4)), '倍；不是年度预测', '计算'],
    ['DCF 起始每股经济现金流代理', F('(B4-B5)*4/B3', owner), 'USD；季度年化并扣股权薪酬', '单季年化及固定股数均为限制'],
]
scenarios = [['情景', 'AI收入', '其他收入', '合计收入', '调整后净利率', '稀释股数', 'EPS', '期末PE', '一年后价格', '不含股息回报', '折现现值']]
for r, (name, ai, other, margin, sh, pe) in enumerate([
    ('悲观',65,50,.48,5,22), ('基准',90,54,.55,5,30), ('乐观',115,57,.58,5,34)
], 2):
    revenue = ai + other
    eps = revenue * margin / sh
    target = eps * pe
    scenarios.append([name,ai,other,F(f'B{r}+C{r}',revenue),margin,sh,F(f'D{r}*E{r}/F{r}',eps),pe,F(f'G{r}*H{r}',target),F(f'I{r}/Inputs!B2-1',target/price-1),F(f'I{r}/(1+Inputs!B6)',target/1.12)])
scenarios.extend([
    ['金额为十亿美元，股数为十亿股，EPS和股价为美元。'],
    ['所有情景输入均为独立假设；乐观AI收入对齐管理层展望，非保证。'],
    ['估值时点约为一年后；无情景概率；不得将目标价当成今天现值。'],
    ['基准现值折价', F('1-Inputs!B2/K3', 1-price/(475.2/1.12))],
    ['30倍PE下要求EPS', F('Inputs!B2*(1+Inputs!B6)/30',price*1.12/30)],
    ['30倍PE下要求AI收入', F('B9*5/0.55-54',(price*1.12/30)*5/.55-54)],
    ['25倍PE下要求AI收入', F('Inputs!B2*(1+Inputs!B6)/25*5/0.55-54',price*1.12/25*5/.55-54)],
])
dcf = [['五年年增长率','第1年现金流','第2年现金流','第3年现金流','第4年现金流','第5年现金流','显式现金流现值','第五年终值','终值现值','总现值','终值占比']]
for r,g in enumerate([.2,.3,.4],2):
    flows = [owner*(1+g)**t for t in range(1,6)]
    pv = sum(cf/1.12**t for t,cf in enumerate(flows,1))
    terminal = flows[-1]*1.04/(.12-.04)
    tpv=terminal/1.12**5
    row=[g]+[F(f'Inputs!B16*(1+A{r})^{t}',cf)for t,cf in enumerate(flows,1)]
    formula='+'.join(f'{chr(65+t)}{r}/(1+Inputs!B6)^{t}' for t in range(1,6))
    row += [F(formula,pv),F(f'F{r}*(1+Inputs!B7)/(Inputs!B6-Inputs!B7)',terminal),F(f'H{r}/(1+Inputs!B6)^5',tpv),F(f'G{r}+I{r}',pv+tpv),F(f'I{r}/J{r}',tpv/(pv+tpv))]
    dcf.append(row)
dcf.extend([['这是简化股权DCF；起始值是单季度经济调整后年化代理，不是已实现年度所有者收益。'],['固定股数；现金流含利息；不重复扣净债务；未计额外担保损失或超额现金分配。'],['年增长率、折现率和永续增长率均为独立模型假设。']])
sources=[['编号','出版者与资料','日期','链接','用途'],
['S1','Stock Analysis historical prices','2026-09-08','https://stockanalysis.com/stocks/avgo/history/','报价'],
['S2','Stock Analysis overview','2026-09-09 accessed','https://stockanalysis.com/stocks/avgo/','股数与市值'],
['S3','Broadcom SEC Q3 exhibit 99.1','2026-09-02','https://www.sec.gov/Archives/edgar/data/1730168/000173016826000076/avgo-08022026x8kxex99.htm','季度数据'],
['S4','Stock Analysis quarterly cash flows','2026-09-09 accessed','https://stockanalysis.com/stocks/avgo/financials/cash-flow-statement/?p=quarterly','TTM FCF'],
['S5','Stock Analysis forecast','2026-09-07 updated','https://stockanalysis.com/stocks/avgo/forecast/','FY2026 EPS consensus'],
['S6','Benzinga Q3 call transcript','2026-09-02 call','https://www.benzinga.com/news/26/09/61592658/broadcom-q3-2026-earnings-call-transcript','管理层AI展望'],
['说明','完整研究与风险来源见同目录报告','','','数字模型假设不是公司指引或独立市场共识']]

def col(n):
    s=''
    while n:
        n,r=divmod(n-1,26)
        s=chr(65+r)+s
    return s

ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
def sheetxml(rows):
    out=[f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="38" customWidth="1"/><col min="2" max="11" width="22" customWidth="1"/></cols><sheetData>']
    for ri,row in enumerate(rows,1):
        out.append(f'<row r="{ri}">')
        for ci,v in enumerate(row,1):
            pos=f'{col(ci)}{ri}'
            if isinstance(v,tuple):
                out.append(f'<c r="{pos}" s="2"><f>{escape(v[0])}</f><v>{v[1]}</v></c>')
            elif isinstance(v,(int,float)):
                out.append(f'<c r="{pos}" s="2"><v>{v}</v></c>')
            else:
                out.append(f'<c r="{pos}" t="inlineStr" s="{1 if ri==1 else 0}"><is><t>{escape(str(v))}</t></is></c>')
        out.append('</row>')
    out.append('</sheetData></worksheet>')
    return ''.join(out)

sheets={'Inputs':inputs,'Scenarios':scenarios,'DCF':dcf,'Sources':sources}
styles=f'''<styleSheet xmlns="{ns}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/><xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs></styleSheet>'''
with zipfile.ZipFile(ROOT/'AVGO_valuation_model.xlsx','w',zipfile.ZIP_DEFLATED) as z:
    overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,5))
    z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+overrides+'</Types>')
    z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(sheets,1))+'</sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>')
    z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'for i in range(1,5))+'<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    z.writestr('xl/styles.xml',styles)
    for i,rows in enumerate(sheets.values(),1):
        z.writestr(f'xl/worksheets/sheet{i}.xml',sheetxml(rows))

print(json.dumps({'report_characters':len(md),'files':[p.name for p in ROOT.iterdir()]},ensure_ascii=False))
