"""Lossless, streaming Excel export; complete pair data remain in CSV.gz/NPZ."""
import csv
import hashlib
import json
import re
from pathlib import Path


def export_workbook(report):
    import xlsxwriter
    report=Path(report);target=report/'research_tables.xlsx'
    temporary=report/'.research_tables.partial.xlsx'
    sources=sorted(p for p in (report/'tables').rglob('*.csv') if 'per_pair' not in p.parts)
    dictionary=[];used=set()
    try:
        with xlsxwriter.Workbook(temporary,{'constant_memory':True,'strings_to_formulas':False,'strings_to_urls':False}) as book:
            heading=book.add_format({'bold':True,'bg_color':'#163B52','font_color':'white','text_wrap':True})
            decimal=book.add_format({'num_format':'0.000000'})
            readme=book.add_worksheet('README')
            notes=[['V5 research tables','Values are not rounded in CSV/NPZ; Excel displays six decimals.'],
                   ['Full pair-level data','See tables/per_pair/*.csv.gz and the original prediction NPZ archives.'],
                   ['Row limits','Large tables are split across sheets, never truncated.'],
                   ['Uncertainty','Outer-split SD and fixed-model user bootstrap are different quantities.'],
                   ['Test boundary','Validation-only workbooks contain no test metrics or test pair data.'],
                   ['Index','All source paths, SHA256 values and sheet names are listed on INDEX.']]
            for n,row in enumerate(notes):readme.write_row(n,0,row)
            readme.set_column(0,0,24);readme.set_column(1,1,105)
            for path in sources:
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                with path.open(newline='',encoding='utf8') as stream:
                    reader=csv.reader(stream);header=next(reader,[])
                    if not header:continue
                    if len(header)>16384:raise ValueError('Excel column limit exceeded: '+str(path))
                    part=0;row_number=1048576;sheet=None;total=0;sheets=[]
                    for row in reader:
                        if any(len(cell)>32767 for cell in row):
                            # Config/phase JSON may exceed Excel's cell limit;
                            # keep it exact in CSV and use an explicit reference.
                            row=[cell if len(cell)<=32767 else '[Full value in '+str(path.relative_to(report))+f'; CSV row {total+2}]' for cell in row]
                        if row_number>=1048576:
                            part+=1
                            stem=re.sub(r'[\[\]:*?/\\]','_',path.stem)[:19]
                            identity=hashlib.sha256(str(path.relative_to(report)).encode()).hexdigest()[:5]
                            name=f'{stem}_{identity}_{part}'[:31]
                            if name in used:raise ValueError('Duplicate workbook sheet '+name)
                            used.add(name);sheets.append(name);sheet=book.add_worksheet(name)
                            sheet.write_row(0,0,header,heading);sheet.freeze_panes(1,0)
                            sheet.set_column(0,len(header)-1,17,decimal)
                            row_number=1
                        values=[]
                        for value in row:
                            try:
                                number=float(value)
                                import math
                                # Preserve large pair IDs as strings rather
                                # than silently losing Excel's 15-digit precision.
                                values.append(number if math.isfinite(number) and not (number.is_integer() and abs(number)>=1e15) else value)
                            except ValueError:values.append(value)
                        sheet.write_row(row_number,0,values);row_number+=1;total+=1
                    if not sheets:
                        identity=hashlib.sha256(str(path.relative_to(report)).encode()).hexdigest()[:12]
                        name=('empty_'+identity);sheet=book.add_worksheet(name);sheet.write_row(0,0,header,heading);sheets=[name]
                    for name in sheets:
                        book.get_worksheet_by_name(name).autofilter(0,0,max(1,min(total,1048575)),len(header)-1)
                    dictionary.append(dict(source=str(path.relative_to(report)),sha256=digest,rows=total,sheets=sheets))
            index=book.add_worksheet('INDEX');index.write_row(0,0,['source','rows','sheets','SHA256'],heading)
            for n,row in enumerate(dictionary,1):index.write_row(n,0,[row['source'],row['rows'],', '.join(row['sheets']),row['sha256']])
            index.freeze_panes(1,0);index.set_column(0,0,45);index.set_column(1,1,12);index.set_column(2,2,45);index.set_column(3,3,68)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    manifest=dict(workbook=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),sources=dictionary,
                  complete_pair_data='tables/per_pair/*.csv.gz; no sampling or truncation',
                  oversized_cells='Explicit CSV references; original full values are retained in CSV')
    (report/'WORKBOOK_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    return manifest
