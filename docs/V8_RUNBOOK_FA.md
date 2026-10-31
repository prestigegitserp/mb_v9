# راهنمای عملی نسخه۸: از نصب تا آزمایش‌های مقاله

هدف این راهنما رفع ابهام TRAIN، VAL و TEST است. پیشنهاد عملی: ابتدا smoke، سپس پایلوت ثابت، سپس مطالعهٔ تنظیم‌شده. بعد از آن و بر اساس هزینه و دامنهٔ مقاله، آزمایش‌های تکمیلی را اجرا کنید. تنظیمات نسخه۷ در فایل‌های خودش محفوظ است؛ خروجی جدید را در پوشهٔ تازهٔ نسخه۸ بنویسید.

## ۱. دریافت و آماده‌سازی

در پوشه‌ای جدا از نسخه۷:

```bash
git clone --branch v8_development --single-branch https://github.com/prestigegitserp/mb_v9.git mb_v9_v8
cd mb_v9_v8
python -m pip install -r requirements_v8.txt
python verify_v8.py
python -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

فایل‌های دیتاست را می‌توان از پوشهٔ نسخه۷ خواند؛ لازم نیست آن‌ها را کپی کنید. مثلاً `--data-root /path/to/mb_v9_v7/datasets`. مسیر ml-100k باید حاوی `ml-100k/u.data` باشد. `--device cuda` در Python بدون CUDA خطا می‌دهد؛ `auto` اجازهٔ انتخاب CPU می‌دهد. fallback بعد از OOM، دستگاه واقعی، allocated/reserved و استفاده از host/SSD همچنان ثبت می‌شوند.

نسخه‌های قبلی را در حین اجرا checkout یا pull نکنید. برای ادامهٔ اجرای قبلی، همان سورس و همان مطالعه را نگه دارید. نسخه۸ برای وزن‌ها و خروجی‌های جدید پوشهٔ مستقل دارد؛ checkpoint نسخه۷ را به‌عنوان checkpoint نسخه۸ جا نمی‌زند.

در Jupyter، برای نصب در همان کرنل:

```python
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_v8.txt'], check=True)
```

پس از نصب PyTorch یا تغییر محیط، در صورت نیاز کرنل را restart کنید. نوت‌بوک آماده در `notebooks/V8_Research_Workflow_FA.ipynb` است.

## ۲. رابطهٔ مراحل و نقش داده

| مرحله | دادهٔ آموزش/انتخاب | TEST | خروجی اصلی |
|---|---|---|---|
| smoke | آموزش بسیار کوتاه، انتخاب VAL | دارد | سلامت فنی؛ شکل‌ها watermark دارند |
| pilot | تنظیمات پیش‌فرض، انتخاب epoch و postfit با VAL | دارد | پایلوت ثابت در `main/` |
| scan | آموزش TRAIN، انتخاب config/K/init با VAL | ندارد | `search/validation_winners.csv` |
| test یا final | بازآموزی config منجمد منتخب هر split | دارد | جدول و نمودار اصلی TEST |
| develop | scan و سپس test خودکار | دارد | مجموعهٔ تنظیم‌شدهٔ کاملِ اصلی |
| lossstudy | تغییر یک ضریب حول winner؛ فقط VAL | ندارد | حساسیت ضرایب |
| convergence | بازآموزی با اندازه‌گیری TRAIN/VAL | ندارد | منحنی همگرایی با schedule ثابت |
| ablation | حذف مؤلفه، با تنظیمات منجمد | دارد | تحلیل سهم مؤلفه؛ TEST توصیفی |
| calibration | انتخاب متد۷ با VAL NLL | دارد | آزمایش احتمالاتی جدا از RMSE اصلی |
| report | فقط بازسازی از فایل‌های موجود | TEST موجود را گزارش می‌کند | Excel، شکل و CSV |
| status | فقط خواندن وضعیت فایل‌ها | آموزش نمی‌دهد | `MANUSCRIPT_CHECKLIST.md/json` |

TEST را در هر epoch اندازه نمی‌گیریم. منحنی همگرایی باید TRAIN/VAL باشد؛ نمودارهای مقالهٔ TEST، مقایسهٔ خروجی‌های نهایی، توزیع خطا، زیرگروه‌ها، کران، هزینه و احتمال‌اند. نداشتن منحنی TEST در طول آموزش نقص نیست.

## ۳. سطح اول: smoke کوتاه

```bash
python paper_v8.py --stage smoke --profile common --datasets ml-100k --data-root ./datasets --output ./results_v8/smoke --device cuda
python paper_v8.py --stage status --output ./results_v8/smoke
```

این مرحله `main --quick` را اجرا می‌کند؛ هشت مدل و baselineهای مجاز همان دیتاست آموزش کوتاه می‌بینند و TEST هم ارزیابی می‌شود. نتیجهٔ آن برای سنجش دقت مقاله مناسب نیست. در synthetic، بعضی baselineها طبق جدول سازگاری قدیمی اجرا نمی‌شوند؛ این حذف یک متد شما نیست.

اگر فقط سلامت اجرا لازم است و نمی‌خواهید زمان رسم شکل صرف شود:

```bash
python paper_v8.py --stage smoke --profile common --datasets ml-100k --data-root ./datasets --output ./results_v8/smoke_no_plots --device cuda --no-plots
python paper_v8.py --stage report --output ./results_v8/smoke_no_plots --plots
```

## ۴. سطح دوم: پایلوت ثابت با TEST

```bash
python paper_v8.py --stage pilot --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/pilot --device cuda
```

بودجهٔ light همان سقف‌های سبکِ قابل مقایسه با اجرای قبلی است: mean epochs=120، steps=2، baseline epochs=200 و سقف search=6 برای مراحل جست‌وجو. در pilot جست‌وجوی config انجام نمی‌شود؛ فقط epoch و postfit با VAL انتخاب می‌شوند. یک split و یک seed با K=20 پیش‌فرض است.

مسیر خروجی pilot:

```text
results_v8/pilot/ml-100k/main/
  locked_test_runs.csv
  locked_test_summary.csv
  publication/research_tables.xlsx
  publication/figures/selected_test_rmse.png
  predictions/
  selected_models/
```

این جدول TEST واقعی است. بااین‌حال، پایلوت یک‌splitی شاهد توصیفی است.

## ۵. سطح سوم: تنظیم هایپرپارامتر و سپس TEST

ابتدا فقط طرح و هزینهٔ بیرونی را ببینید:

```bash
python paper_v8.py --stage plan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/tuned_light --device cuda
```

سپس دو مرحلهٔ روشن:

```bash
python paper_v8.py --stage scan --output ./results_v8/tuned_light
python paper_v8.py --stage test --output ./results_v8/tuned_light
python paper_v8.py --stage status --output ./results_v8/tuned_light
```

recipe مرحلهٔ plan ذخیره شده است؛ فرمان‌های بعدی profile، data-root، seedها، K و سقف‌ها را خودکار بازیابی می‌کنند. اگر plan را اجرا نکرده‌اید، همهٔ تنظیمات اولیه را به scan بدهید:

```bash
python paper_v8.py --stage scan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/tuned_light --device cuda
```

راه یک‌فرمانیِ همین مسیر:

```bash
python paper_v8.py --stage develop --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/tuned_auto --device cuda
```

`develop` ابتدا scan و سپس TEST می‌گیرد. در نسخه۷ نیز قرار بود همین کار انجام شود، ولی باگ هویت initialization در finalِ MBMF مانع می‌شد. در نسخه۸ هویت مؤثر هر MBMF کاندیدا همان init منتخب است؛ سایر تنظیمات و جمعیت داده همچنان سخت‌گیرانه کنترل می‌شوند.

### فایل‌های برنده کجا هستند؟

- `search/hyperparameter_validation_all.csv`: همهٔ آموزش‌های جست‌وجو، بدون ستون TEST.
- `search/validation_winners.csv`: برندهٔ هر روش برای هر outer split.
- `search/best_configurations.json`: تنظیمات برندگان و postfit هر seed.
- `search/publication/tables/search_parameter_coverage.csv`: چه پارامتری واقعاً چند مقدار داشته است.
- `search/publication/tables/search_budget.csv`: تعداد آموزش و زمان واقعی.
- `search/publication/tables/hyperparameter_sensitivity.csv`: نقاط config کامل؛ اثر مستقل و علّی یک ضریب نیست.

فایل best_configurations ورودی مستقیم `--v8-config` نیست. برای مصرف خودکار آن از `test/final` استفاده کنید. `main` پیش‌فرض‌ها را اجرا می‌کند، نه winnerهای search.

### چه چیزی را در همان پوشه تغییر ندهم؟

پروفایل، seed، K، روش‌های مطالعه و بودجهٔ علمی را در همان پوشه عوض نکنید. برنامه تغییر صریح ناسازگار را رد می‌کند. برای مطالعهٔ تازه خروجی تازه بسازید. محل اجرا و منابع مانند CPU/GPU، threads، host و SSD قابل تنظیم‌اند، ولی دستگاه واقعی و محدودیت مقایسه باید در گزارش بماند. جابه‌جایی بین CPU و GPU تضمین برابری بیتی ندارد.

مثلاً ادامه پس از توقف:

```bash
python paper_v8.py --stage scan --output ./results_v8/tuned_light
python paper_v8.py --stage test --output ./results_v8/tuned_light
```

resume در مراحل آموزش خودکار فعال است. متد۸ قابلیت بازیابی کاندیدای تمام‌شده را حفظ می‌کند؛ بازیابی وسط optimizer آن مانند هفت متد دیگر ادعا نمی‌شود.

## ۶. نمودارهای TEST چگونه ساخته می‌شوند؟

پس از موفقیت test:

```text
results_v8/tuned_light/ml-100k/
  final_locked_test_all.csv
  final_locked_test_summary.csv
  final_selection_provenance.json
  MANUSCRIPT_CHECKLIST.md
  publication/research_tables.xlsx
  publication/figures/
  publication/tables/
  final/Method1/split_2026/predictions/
  ...
```

شکل‌های اصلی شامل RMSE و MAE، صدک‌های خطا، ECDF و residual، خطا برحسب rating و تعداد مشاهدات، کالیبراسیون نقطه‌ای، کران‌ها، هزینه/دقت، زمان و حافظه است. در common، متد۷ می‌تواند NLL/RPS، reliability، PIT، پوشش interval و risk–coverage هم داشته باشد. هر شکل در PNG با۴۰۰dpi، PDF برداری و SVG قابل ویرایش ذخیره می‌شود.

شکل‌های داخل `search/` مربوط به VAL هستند. شکل `publication/figures/selected_test_rmse.png` در ریشهٔ دیتاست مربوط به TEST نهایی است. در گزارش هر پوشه، `PAPER_OUTPUTS.md`، primary partition را صریح نوشته است. فایل‌های خام هر روش حفظ می‌شوند؛ برای جلوگیری از تکرار ده‌ها شکل یکسان، گزارش اصلی final به‌صورت تجمیعی رسم می‌شود.

برای بازسازی ظاهر بدون آموزش:

```bash
python paper_v8.py --stage report --output ./results_v8/tuned_light --plots
```

برای یک شکل سفارشی از CSV:

```bash
python replot_v8.py --input ./results_v8/tuned_light/ml-100k/publication/tables/metric_summary.csv --where metric=test_RMSE --x method --y mean --kind barh --output ./results_v8/custom_test_rmse.png
```

فایل داده و SHA ورودی شکل سفارشی ذخیره می‌شود. report نتیجهٔ TEST گمشده را جعل یا از VAL جایگزین نمی‌کند.

## ۷. ضرایب لاس و آزمایش تک‌عاملی

```bash
python paper_v8.py --stage lossstudy --output ./results_v8/tuned_light
```

این مرحله فقط VAL است و می‌تواند پرهزینه باشد. برای شروع هدفمند:

```bash
python paper_v8.py --stage lossstudy --output ./results_v8/tuned_light --loss-methods Method6 --loss-parameters transport_moment_weight
```

خروجی‌ها: `lossstudy/loss_coefficient_validation.csv`، `loss_coefficient_summary.csv`، در صورت احتمال `loss_coefficient_probability_summary.csv`، و `loss_sensitivity.xlsx`. ضریب فعال جهت متد۷ این بار در registry وجود دارد. وزن غیرفعال mechanism برای متد۷ نمودار حساسیت ساختگی نمی‌سازد.

lossstudy winner اصلی را خودکار تغییر نمی‌دهد. اگر از آن برای انتخاب مدل تازه استفاده کنید، این تصمیم بخشی از توسعه است؛ نتیجهٔ تأییدی بعدی باید مستقل از انتخاب روی TEST باشد. متد۸ تابع هدف اصلی خود را حفظ می‌کند و برای آن لاس کمکی ساختگی اضافه نمی‌شود.

## ۸. تحلیل سازوکار و پایداری

```bash
python paper_v8.py --stage ablation --output ./results_v8/tuned_light
python paper_v8.py --stage ablation-fixed --output ./results_v8/tuned_light
python paper_v8.py --stage convergence --output ./results_v8/tuned_light
python paper_v8.py --stage sparsity --output ./results_v8/tuned_light
python paper_v8.py --stage kstudy --output ./results_v8/tuned_light
```

تعداد ablationها از تنظیمات واقعی winner مشتق می‌شود؛ عدد ثابت۳۸ به‌عنوان شرط صحت تحمیل نمی‌شود. حذف گیت اعتماد، حذف کالیبراسیون هندسه، پاسخ گسسته و decoder-only نیز کنترل‌های مشخص دارند. تنظیم‌شدن ضرایب در ablation با TEST انجام نمی‌شود.

در recipe فعلی K فقط۲۰ است؛ اجرای kstudy تنها یک نقطه می‌دهد. برای مطالعهٔ حساسیت چند K، از ابتدا در مطالعهٔ تازه `--k 10,20,50` را مشخص کنید. K اصلی از VAL انتخاب می‌شود؛ منحنی TEST برای انتخاب دوبارهٔ بهترین K نیست.

sparsity بخشی از TRAIN را حذف و VAL/TEST را حفظ می‌کند. این آزمایش با کاربران یا آیتم‌های کاملاً دیده‌نشده یکسان نیست. convergence دوباره آموزش می‌دهد و RMSE کامل TRAIN را با schedule checkpoint قبلی می‌گیرد؛ نمودار train از loss masked متفاوت است.

## ۹. احتمال و split زمانی

برای probability متد۷ در پروفایل common:

```bash
python paper_v8.py --stage calibration --output ./results_v8/tuned_light
```

این انتخاب با VAL NLL است و باید جدا از مقایسهٔ اصلی RMSE برچسب بخورد. تغییر پوشش در risk–coverage مسئلهٔ ارزیابی را تغییر می‌دهد؛ عدد کمتر پس از حذف نمونه‌ها، RMSE تمام TEST نیست.

```bash
python paper_v8.py --stage temporal --output ./results_v8/tuned_light
```

temporal فقط برای MovieLens است و scan/final مستقل خود را دارد. نتیجهٔ آن را با TEST تصادفی به‌عنوان یک cohort یکسان جمع نکنید.

## ۱۰. مطالعهٔ بزرگ مقاله

پروفایل common: تقسیم per-user حدود۸۰/۱۰/۱۰ و نقش‌های disjoint انتخاب/برازش احتمال/holdout. در ML-100K/2026، TRAIN=80053، VAL اصلی=9967 و TEST=9980 است؛ selection حدود۶۰٪ VAL را دارد.

پروفایل paper/development برای دادهٔ واقعی: پروتکل aemc-paper با نسبت۸۵٫۵/۴٫۵/۱۰ و تمام VAL برای انتخاب؛ متد۷ point-only است. این یک تغییر پروتکل است، نه اثبات بهبود نسخه. برای پژوهش احتمال، یک مطالعهٔ common جدا لازم است.

ابتدا مطالعهٔ میانیِ تازه:

```bash
python paper_v8.py --stage plan --profile paper --datasets ml-100k --data-root ./datasets --output ./results_v8/paper_medium --device cuda --split-seeds 4101,4102,4103 --model-seeds 11,22 --k 10,20
python paper_v8.py --stage scan --output ./results_v8/paper_medium
python paper_v8.py --stage test --output ./results_v8/paper_medium
python paper_v8.py --stage ablation --output ./results_v8/paper_medium
python paper_v8.py --stage lossstudy --output ./results_v8/paper_medium
python paper_v8.py --stage status --output ./results_v8/paper_medium
```

برای پروفایل بزرگ پیش‌فرض۲۰ split، سه seed مدل و Kهای۱۰،۲۰،۵۰:

```bash
python paper_v8.py --stage plan --profile paper --datasets ml-100k --data-root ./datasets --output ./results_v8/paper_full --device cuda
python estimate_v8.py --profile paper --dataset ml-100k
python paper_v8.py --stage all --output ./results_v8/paper_full --include-temporal
```

این اجرای بسیار بزرگی است. تخمین زمان صرفاً heuristic بر مبنای مرجع ثبت‌شده است و بنچمارک RTX3050 یا سخت‌افزار شما نیست. `--budget full` سقف تعداد jobهای گرید تعریف‌شده را برمی‌دارد؛ تضمین جست‌وجوی تمام فضای ممکن یا رسیدن به بهینهٔ جهانی نمی‌دهد. برای همهٔ داده‌ها `--datasets all` بگذارید، ولی ابتدا plan و ظرفیت منابع را بررسی کنید.

## ۱۱. کافی‌بودن برای مقاله چه معنایی دارد؟

پوشش فنی پیشنهادی: جدول اصلی TEST با baselineهای تنظیم‌شده، چند split/seed، چند دیتاست متناسب با ادعا، trained ablation، کنترل decoder و postprocessing، حساسیت ضرایب روی VAL، هزینه و حافظه، و robustness متناسب با دامنه. برای ادعای uncertainty، baseline احتمالاتی و ارزیابی proper scoring ruleها نیز لازم است.

این مجموعهٔ کد امکان تولید این شواهد را فراهم می‌کند، اما novel بودن، قدرت baselineهای روز، کفایت اثبات نظری یا پذیرش Q1 را تأیید خودکار نمی‌کند. فایل checklist دقیقاً برای جداکردن «وجود خروجی» از «ادعای علمی» است. یک split یا seedهای آموزش، نمونه‌های مستقل از کل مسئله محسوب نمی‌شوند؛ هم‌پوشانی resplitها نیز باید در تفسیر آزمون آماری لحاظ شود.
