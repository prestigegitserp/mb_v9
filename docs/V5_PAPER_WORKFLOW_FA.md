# اجرای مقاله با نسخهٔ ۵

این راهنما برای تغییرات V5 است. این تغییرات فعلاً commit/merge نشده‌اند؛ دریافت شاخهٔ remote پیش از انتشار کد، همان پایهٔ V4 را می‌دهد. اجرای دستورها مستلزم checkout دارای فایل‌های V5 است. نمونه‌ها با Python در Windows و Linux قابل اجرا هستند؛ در VS Code همان interpreter محیط نصب را انتخاب کنید.

## ۱. بررسی نصب و دستگاه

```bash
python -m pip install -r requirements_v5.txt
python verify_v5.py
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

نسخهٔ CUDAدار PyTorch باید با دستگاه شما سازگار باشد. خروجی `sys.executable` باید همان محیط VS Code باشد. `--device cuda:0` در نبود CUDA صریحاً متوقف می‌شود؛ `auto` دستگاه قابل استفاده را انتخاب می‌کند. برای آزمون حافظه روی **کارت خودتان**:

```bash
python experiments/v5_execution_parity.py --device cuda:0 --readout geometric --steps 3 --threads 2 --output ./parity_geometric --cache-dir ./v5_ssd_cache
python experiments/v5_execution_parity.py --device cuda:0 --readout coupled --steps 3 --threads 2 --output ./parity_coupled --cache-dir ./v5_ssd_cache
python experiments/v5_execution_parity.py --device cuda:0 --readout decoder --steps 3 --threads 2 --output ./parity_decoder --cache-dir ./v5_ssd_cache
```

این دستورها از TRAIN مصنوعی، ماسک ثابت و مقایسهٔ وزن و AdamW استفاده می‌کنند. `EXECUTION_PARITY.json` باید `all_bitwise_equal=true` باشد. برای بررسی CPU فقط دستگاه را به `cpu` تغییر دهید؛ نتیجهٔ آن تأیید CUDA نیست.

سیاست پیش‌فرض `auto` تا زمان وجود فضای آزاد کافی گراف را روی CUDA نگه می‌دارد؛ سپس saved tensorها به RAM و در صورت رسیدن به بودجه به SSD منتقل می‌شوند. روی CPU هم در رسیدن به حد ذخیرهٔ RAM، saved tensorها به SSD می‌روند. این انتقال همهٔ داده و optimizer state را شامل نمی‌شود. فایل‌های موقت پس از candidate پاک می‌شوند. پس از OOM، candidate از ابتدا با همان وزن/RNG روی سیاست کم‌حافظه‌تر و در نهایت CPU بازاجرا می‌شود. update ناقص، کاهش batch، تغییر precision و نرخ یادگیری جایگزین نمی‌شوند. overflow تکرارشونده روی CPU خطای صریح است.

در ورودی مستقیم، گزینه‌های `--v5-host-budget-mb 2048`، `--v5-ssd-budget-gb 64`، `--v5-cache-dir ./v5_ssd_cache` و `--v5-memory-reserve 0.15` قابل تنظیم‌اند. `--no-v5-cpu-fallback` برای الزام ادامهٔ اجرا فقط روی دستگاه انتخاب‌شده است. NVML device-wide است؛ درصد فعالیت آن معادل FLOPs مدل یا تضمین حداکثر سرعت نیست.

## ۲. طرح و smoke

```bash
python paper_v5.py --phase plan --dataset ml-100k --output ./paper_v5
python paper_v5.py --phase smoke --dataset ml-100k --output ./paper_v5 --device cuda:0
```

launcher تنظیمات ML-100K را از `examples/v5/recommended_v5_ml100k.json` می‌خواند. سه split برابر3031،3032،3033، سه model seed برابر11،22،33 و Kهای10،20،50 دارد. طرح کامل چند هزار fit دارد؛ `plan` تعداد واقعی را پیش از آموزش چاپ می‌کند. هزینهٔ OOF و بازاجراها اضافه است. برای طرح آزمایشی محدود می‌توانید همهٔ مراحل را با `--split-seeds 3031 --model-seeds 11 --k 20 --search-jobs-per-method 4` اجرا کنید؛ این grid ناقص را نتیجهٔ جست‌وجوی کامل نام‌گذاری نکنید.

`smoke` در زیرپوشهٔ مستقل `smoke` و با بودجهٔ بسیار کوچک اجرا می‌شود. کیفیت آن برای مقاله نیست. برای دیدن دستور launcher بدون آموزش، `--dry-run` اضافه کنید. برای مقایسه‌های اصلی پیش از مشاهدهٔ TEST، روش‌ها، seedها، بودجه، metric و قاعدهٔ انتخاب را ثابت کنید. split جدیدِ همان دیتاست استقلال از تمام آزمایش‌های قبلی را تضمین نمی‌کند.

## ۳. اسکن هایپرپارامتر؛ پاسخ به سؤال اصلی

```bash
python paper_v5.py --phase scan --dataset ml-100k --output ./paper_v5 --device cuda:0
```

**بله، اسکن جدول و نمودار، پیش‌بینی TRAIN/VAL، تاریخچه، تنظیمات، فایل‌های تشخیصی و Excel را تولید می‌کند.** گزارش هر job و گزارش تجمیعی نوشته می‌شوند. نمودار TEST در validation-only نباید ساخته شود. `scan` این launcher عمداً پیش از TEST می‌ایستد. ورودی مستقیم `python run_v5.py scan ...` بدون `--scan-only` پس از انتخاب به final ادامه می‌دهد.

انتخاب config/K/initialization با میانگین validation روی model seedهای اعلام‌شده و جدا برای هر روش/outer split انجام می‌شود. TEST در جدول انتخاب نیست. برای fairness، همهٔ روش‌ها selection یکسان دارند. در common protocol، روش۷ علاوه بر selection از دو نقش جدا برای برازش/کنترل شکل احتمال استفاده می‌کند؛ مصرف این داده در مقاله باید صریح نوشته شود. میانگین خروجی پس از probability fitting ثابت می‌ماند.

`--v5-retain-candidates metrics` وزن کاندیداهای موقت را حذف می‌کند؛ CSV، prediction، تاریخچه و گزارش باقی می‌مانند. وزن نهایی منتخب در final ذخیره می‌شود. `--v5-no-plots` رسم را عقب می‌اندازد و داده‌ها را نگه می‌دارد؛ بعداً `report` بزنید. launcher به‌طور پیش‌فرض رسم را فعال نگه می‌دارد. ذخیرهٔ تمام predictionها و per-pairها به دیسک قابل توجهی نیاز دارد.

## ۴. پیوند اسکن به آزمایش اصلی

```bash
python paper_v5.py --phase final --dataset ml-100k --output ./paper_v5 --device cuda:0
```

همان output و دقیقاً همان config، seedها، K، protocol و grid مرحلهٔ scan را نگه دارید. `final` برندگان `search/validation_winners.csv` را می‌خواند، config و امضای داده/سورس را کنترل می‌کند و همان مدل‌های ثابت‌شده را بازآموزی و ارزیابی می‌کند. تغییر خودسرانهٔ تنظیمات یا config اسکن باعث توقف است. validation به TRAIN اضافه نمی‌شود. خروجی TRAIN/VAL/TEST و checkpoint منتخب ذخیره می‌شوند. `main` پیش‌فرض‌های همان اجرا را می‌آزماید و اسکن پوشهٔ دیگری را خودکار وارد نمی‌کند.

## ۵. ابلیشن، K، کمبود داده، calibration و همگرایی

```bash
python paper_v5.py --phase all --dataset ml-100k --output ./paper_v5 --device cuda:0
```

کارهای کامل‌شده با `resume` بازاستفاده می‌شوند؛ سپس مراحل تکمیلی اجرا می‌شوند. `all` شامل convergence هم هست و بسیار پرهزینه است. برای اجرای جدا، `--phase` را به `ablation`، `ablation-fixed`، `kstudy`، `sparsity`، `calibration` یا `convergence` تغییر دهید و تمام آرگومان‌های مطالعه را نگه دارید.

| مرحله | تنظیمات و تفسیر |
|---|---|
| ablation | از برندهٔ اسکن همان split؛ حذف مکانیسم با K و سایر تنظیمات ثابت |
| ablation-fixed | full و حذف‌ها با K یکسان از پیش اعلام‌شده؛ تحلیل ثانویه |
| kstudy | سایر تنظیمات منتخب ثابت؛ TEST نمودار توصیفی است، معیار انتخاب K نیست |
| sparsity | حذف فقط TRAIN با VAL/TEST ثابت؛ تعداد واقعی یال‌ها ثبت می‌شود |
| calibration | تحلیل ثانویهٔ روش۷؛ انتخاب با NLL از آزمایش اصلی RMSE جداست |
| convergence | بازاجرای validation-only؛ TEST در هر epoch خوانده نمی‌شود |

ابلیشن‌هایی که config را عوض نمی‌کنند حذف می‌شوند. اگر حذف بازنمایی، readout وابسته را هم حذف کند، این تغییر مرکب در provenance ثبت می‌شود؛ آن را اثر یک ضریب مستقل ننویسید. تعداد پارامتر، زمان، توقف زودهنگام و سقف بودجه همراه خطا گزارش شوند. `run_v5.py all --include-temporal ...` برای MovieLens مطالعهٔ زمانی مستقل را نیز اضافه می‌کند؛ launcher فعلی این گزینه را به‌صورت پیش‌فرض نمی‌فرستد.

## ۶. مقایسهٔ منصفانه و بازتولید مقاله دو پروتکل‌اند

مقایسهٔ اصلی همهٔ روش‌ها با `--protocol common` است. برای تشخیص فاصلهٔ AEMC با مقاله، پوشه‌ای **جدا**، دو baseline مشخص، تمام validation برای انتخاب، بیست split و نسبت‌های خود مقاله استفاده کنید. مثال زیر فقط plan است؛ پس از بررسی هزینه همین آرگومان‌ها را برای scan و final نگه دارید:

```bash
python paper_v5.py --phase plan --dataset ml-100k --methods AutoRec,AEMC-NE --protocol aemc-paper --split-seeds 4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012,4013,4014,4015,4016,4017,4018,4019,4020 --model-seeds 1 --k 20 --output ./aemc_paper_diagnostic
```

`autorec-paper` نسبت81/9/10 دارد؛ مقالهٔ اصلی AutoRec اصلاً عدد ML-100K گزارش نمی‌کند. این presetها به‌تنهایی برابری با کد نویسنده را ثابت نمی‌کنند. Method7 به‌علت نیاز به نقش‌های جدا در این پروتکل‌ها مجاز نیست. `BASELINE_FIDELITY.csv/json` وضعیت تأییدنشده و منابع اصلی را ثبت می‌کند. به‌جای جابه‌جایی مقیاس score، initialization، فعال‌ساز، regularization، objective، optimizer و split را با کد نویسنده تطبیق دهید. شکاف AEMC فعلاً مسئلهٔ باز است.

## ۷. خروجی‌ها و بازسازی بدون آموزش

در جدول زیر `ROOT=paper_v5/ml-100k` است؛ هر job نیز گزارش مستقل دارد.

| مسیر | کاربرد |
|---|---|
| `search/hyperparameter_validation_all.csv` | همهٔ تلاش‌ها؛ بدون TEST |
| `search/validation_winners.csv` و `validation_selection_manifest.json` | انتخاب ثابت‌شده و قاعدهٔ آن |
| `search/publication/research_tables.xlsx` | Excel اسکن، حساسیت و پارامترها |
| `final_locked_test_all.csv` و `final_locked_test_summary.csv` | آزمایش اصلی |
| `publication/research_tables.xlsx` | workbook نهایی با INDEX شامل منبع، hash و تعداد ردیف |
| `publication/tables/metrics_long.csv` | معیارها در قالب long؛ مناسب pivot و نمودار تازه |
| `publication/tables/per_pair/*.csv.gz` | تمام زوج‌ها با truth/prediction/residual؛ بدون نمونه‌برداری |
| `publication/tables/entity_errors.csv` | خطای هر user/item، تعداد، bias و MAE/RMSE |
| `publication/tables/error_tail_summary.csv` | دنبالهٔ خطا و quantileها |
| `publication/tables/conditional_errors.csv` | خطا برحسب rating و پشتیبانی |
| `publication/tables/epoch_history.csv` | تاریخچهٔ TRAIN/VAL؛ masked query loss جداست |
| `publication/tables/training_budget.csv` | بهترین epoch، توقف، update و budget-hit |
| `publication/tables/candidate_resources.csv` | دستگاه و offload/fallback هر candidate |
| `publication/tables/hyperparameters_long.csv` در گزارش search | مقادیر configهای کامل برای حساسیت |
| `figures/` و `publication/figures/` | PNG با400DPI، PDF برداری و SVG قابل ویرایش |
| `predictions/` و `selected_models/` داخل jobها | NPZها، احتمال‌های موجود و وزن منتخب |
| `mechanisms/*/DEVICE_AND_STORAGE.json` | اجرای واقعی، device، cache و بازاجراها |

Excel جدول‌های پرحجم per-pair را به CSV.gz ارجاع می‌دهد؛ داده حذف یا sample نمی‌شود. جدول‌های بزرگ به sheetهای متعدد تقسیم می‌شوند؛ مسیر source، SHA-256 و شمار ردیف در INDEX است. رشته‌های بسیار طولانی با ارجاع صریح به CSV کامل جایگزین می‌شوند. معیار ناموجود خالی می‌ماند؛ برای مدل فاقد توزیع، NLL ساختگی تولید نمی‌شود.

```bash
python paper_v5.py --phase report --dataset ml-100k --output ./paper_v5
python paper_v5.py --phase audit --dataset ml-100k --output ./paper_v5
python replot_v5.py --input ./paper_v5/ml-100k/publication/research_tables.xlsx --sheet metrics_long --where metric=test_RMSE --x method --y value --kind barh --aggregate mean --title "MovieLens-100K" --output ./paper_figures/rmse.png
python replot_v5.py --input ./paper_v5/ml-100k/publication/tables/conditional_errors.csv --where partition=test --where dimension=truth --x group --y RMSE --group method --kind line --output ./paper_figures/error_by_rating.png
```

`replot_v5.py` همچنین scatter، box و heatmap دارد. برای heatmap، `--x` و `--y` دو محور دسته‌بندی و `--value` ستون عددی است؛ `--aggregate mean` یا median تکرارها را خلاصه می‌کند. هر نمودار تازه همراه CSV دقیق دادهٔ ورودی رسم، JSON منبع و hash، PNG/PDF/SVG ذخیره می‌شود. این دستورها آموزش مجدد ندارند.

شکل‌های رتبه‌بندی، ECDF، خطای شرطی، calibration، بودجه/زمان، حساسیت و منحنی‌های چندپنلی آماده‌اند. عدد خطای میله‌ها در رتبه‌بندی SD میان outer splitهاست؛ با یک split خطای آماری مستقل ادعا نمی‌شود. برای استنتاج مقاله، seedهای مدل را نمونهٔ مستقل افراد نگیرید، همپوشانی splitها و انتخاب‌های متعدد را لحاظ کنید و محدودیت دادهٔ قبلاً مشاهده‌شده را گزارش دهید.
