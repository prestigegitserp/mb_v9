"""Create the runnable Persian V8 guide without executing experiments."""
from pathlib import Path
import json


def cell(kind,text):
    item=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
    if kind=='code':item.update(execution_count=None,outputs=[])
    return item


def main():
    root=Path(__file__).resolve().parents[1]
    cells=[
      cell('markdown','# نسخه۸: از پایلوت تا TEST مقاله\nاین نوت‌بوک از paper_v8.py استفاده می‌کند. تنظیم مطالعه ذخیره می‌شود؛ پس از scan، دستور test همان تنظیمات را بازیابی می‌کند. سلول‌های آموزش با False تحویل داده شده‌اند تا اجرای همهٔ سلول‌ها ناخواسته مطالعهٔ سنگین آغاز نکند. راهنمای کامل: docs/V8_RUNBOOK_FA.md. معادل کامل راهنمای نسخه۷، همراه PowerShell و سه سطح اجرا: docs/V8_RELEASE_RUNBOOK_FA.md. معنی مراحل: docs/V8_COMMAND_REFERENCE_FA.md. شاخهٔ انتشار v8 و شاخهٔ توسعه v8_development است.\n'),
      cell('code', '''from pathlib import Path
import sys, subprocess, json
import pandas as pd

PROJECT = Path.cwd()
if PROJECT.name == 'notebooks':
    PROJECT = PROJECT.parent
assert (PROJECT/'paper_v8.py').is_file(), 'Set PROJECT to the repository root.'
DATA_ROOT = PROJECT/'datasets'  # May point to the old V7 datasets directory.
DEVICE = 'cuda'  # Use 'cpu' or 'auto' if desired.
TUNED = PROJECT/'results_v8/tuned_light'

def command(*args):
    return subprocess.run([sys.executable,*map(str,args)],cwd=PROJECT,check=True)

def paper(stage, output, *extra, initial=False):
    args=['paper_v8.py','--stage',stage,'--output',str(output)]
    if initial:
        args += ['--profile','common','--budget','light','--datasets','ml-100k',
                 '--data-root',str(DATA_ROOT),'--device',DEVICE]
    return command(*args,*extra)

command('verify_v8.py')
'''),
      cell('markdown','## ۱. smoke: TEST کوتاه برای سلامت فنی\nشکل‌ها watermark دارند؛ این عددها نتیجهٔ دقت مقاله نیستند.\n'),
      cell('code',"RUN_SMOKE = False\nif RUN_SMOKE:\n    paper('smoke',PROJECT/'results_v8/smoke',initial=True)\n"),
      cell('markdown','## ۲. پایلوت ثابت: TRAIN/VAL و TEST دارد\nconfigهای پیش‌فرض را می‌سنجد؛ جست‌وجوی config انجام نمی‌دهد. خروجی زیر main است.\n'),
      cell('code',"RUN_PILOT = False\nif RUN_PILOT:\n    paper('pilot',PROJECT/'results_v8/pilot',initial=True)\n"),
      cell('markdown','## ۳. طرح مطالعهٔ تنظیم‌شده\nاین مرحله آموزش نمی‌دهد؛ recipe را ذخیره می‌کند. مراحل بعدی تنها output را لازم دارند. برای تغییر علمی مطالعه، پوشهٔ تازه بسازید.\n'),
      cell('code',"RUN_PLAN = False\nif RUN_PLAN:\n    paper('plan',TUNED,initial=True)\n"),
      cell('markdown','## ۴. جست‌وجو فقط روی VAL\nاگر plan را نگرفته‌اید، initial=True را نگه دارید؛ در ادامهٔ مطالعه نیز تنظیمات صریح یکسان معتبرند. هنوز TEST تولید نمی‌شود.\n'),
      cell('code',"RUN_SCAN = False\nif RUN_SCAN:\n    paper('scan',TUNED,initial=True)\n"),
      cell('markdown','## ۵. این سلول TEST واقعی را می‌گیرد\nاین همان final برندگان scan است. هر هشت روش شما و baselineهای مجاز مطالعه حضور دارند؛ فهرست روش‌ها را برای دورزدن خطا حذف نمی‌کنیم.\n'),
      cell('code',"RUN_TEST = False\nif RUN_TEST:\n    paper('test',TUNED)\n"),
      cell('code', '''base=TUNED/'ml-100k'
f=base/'final_locked_test_summary.csv'
if f.is_file():
    table=pd.read_csv(f)
    display(table[[c for c in ['method','test_RMSE_mean','test_RMSE_std','test_MAE_mean','n_outer_splits'] if c in table]])
else:
    print('TEST has not completed:',f)
image=base/'publication/figures/selected_test_rmse.png'
if image.is_file():
    from IPython.display import Image,display
    display(Image(filename=str(image)))
'''),
      cell('markdown','## ۶. گزارش و شکل‌ها بدون آموزش مجدد\nاگر TRAIN را با no-plots گرفته‌اید، plots اینجا آن را فعال می‌کند. از دادهٔ VAL، TEST ساختگی تولید نمی‌شود.\n'),
      cell('code',"RUN_REPORT = False\nif RUN_REPORT:\n    paper('report',TUNED,'--plots')\nRUN_STATUS = False\nif RUN_STATUS:\n    paper('status',TUNED)\n"),
      cell('markdown','## ۷. حساسیت ضریب transport متد۶ فقط با VAL\nاین مرحله تنظیم برنده را خودکار عوض نمی‌کند؛ آزمایش تک‌عاملی است. برای همهٔ ضرایب، دو گزینهٔ فیلتر را حذف کنید.\n'),
      cell('code',"RUN_LOSS_STUDY = False\nif RUN_LOSS_STUDY:\n    paper('lossstudy',TUNED,'--loss-methods','Method6','--loss-parameters','transport_moment_weight')\n"),
      cell('markdown','## ۸. حذف مؤلفه، همگرایی و probability\nablation و sparsity با تنظیمات منجمد و TEST توصیفی‌اند. convergence فقط TRAIN/VAL است. calibration مطالعهٔ جدا با انتخاب VAL NLL برای متد۷ است.\n'),
      cell('code',"RUN_SECONDARY = False\nif RUN_SECONDARY:\n    for stage in ['ablation','ablation-fixed','convergence','sparsity','calibration']:\n        paper(stage,TUNED)\n    paper('audit',TUNED)\n"),
      cell('markdown','## ۹. مطالعهٔ میانی مقاله، در خروجی تازه\nاین مطالعه پروتکل aemc-paper و متد۷ point-only دارد. برای احتمال، common جدا لازم است. تعداد split/seed نمونه‌های مستقل تضمین‌شده نیست. ابتدا فقط plan؛ all ممکن است بسیار پرهزینه باشد.\n'),
      cell('code', '''PAPER = PROJECT/'results_v8/paper_medium'
RUN_PAPER_PLAN = False
if RUN_PAPER_PLAN:
    command('paper_v8.py','--stage','plan','--profile','paper',
            '--datasets','ml-100k','--data-root',DATA_ROOT,'--output',PAPER,
            '--device',DEVICE,'--split-seeds','4101,4102,4103',
            '--model-seeds','11,22','--k','10,20')
RUN_PAPER_SUITE = False
if RUN_PAPER_SUITE:
    paper('all',PAPER,'--include-temporal')
'''),
      cell('markdown','## ۱۰. معادل curves نسخه۷، بدون آموزش\nاین سلول history موجود را رسم می‌کند؛ برای RMSE کامل TRAIN که قبلاً ذخیره نشده، از convergence استفاده کنید.\n'),
      cell('code',"RUN_CURVES = False\nif RUN_CURVES:\n    paper('curves',TUNED,'--plots')\n"),
      cell('markdown','## ۱۱. شمارش برنامه و تخمین زمان با پایلوت دستگاه خودتان\nاین تخمین آموزش نمی‌دهد. مرجع پیش‌فرض از GPU دیگری است؛ برای تخمین مرتبط با دستگاه خودتان، calibration را روی CSV پایلوت همان دستگاه بگذارید.\n'),
      cell('code', '''RUN_ESTIMATE = False
if RUN_ESTIMATE:
    estimate_args=['estimate_v8.py','--profile','common',
                   '--search-jobs-per-method','6','--max-epochs','120',
                   '--steps-per-epoch','2','--baseline-max-epochs','200']
    measured=PROJECT/'results_v8/pilot/ml-100k/main/candidate_validation_runs.csv'
    if measured.is_file():
        estimate_args += ['--calibration',str(measured)]
    command(*estimate_args)
'''),
    ]
    for i,c in enumerate(cells):c['id']=f'v8-guide-{i:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
                      language_info=dict(name='python',version='3')))
    target=root/'notebooks/V8_Research_Workflow_FA.ipynb'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    for c in cells:
        if c['cell_type']=='code':compile(''.join(c['source']),str(target),'exec')
    import nbformat
    nbformat.validate(notebook)
    print(target)


if __name__=='__main__':main()
