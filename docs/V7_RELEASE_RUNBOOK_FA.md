# انتشار ۷.۰.۰: از دریافت تا جدول نهایی مقاله

شاخهٔ انتشار `v7` و شاخهٔ توسعه `v7-development` است. ورودی‌ها `run_v7.py` و `paper_v7.py` هستند؛ `run_v9.py` و VERSION ریشه برای حفظ نسخه‌های قبلی تغییر نکرده‌اند. منبع، معماری و trainer اصلی Method10 با نام عمومی Method8 دست‌نخورده است.

## دریافت در پوشهٔ خالی

در PowerShell داخل پوشهٔ واقعاً خالی:

```powershell
git clone --branch v7 --single-branch https://github.com/prestigegitserp/mb_v9.git .
python -m pip install -r requirements_v7.txt
python verify_v7.py
python -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

بهتر است محیط Python جدا داشته باشید تا نسخهٔ کتابخانه‌های اجرای قبلی حفظ شود. اگر CUDA برابر False بود، نصب GPU متناسب با سیستم را از راهنمای رسمی PyTorch انتخاب کنید؛ `--device cuda` این وضعیت را پنهان نمی‌کند. برای به‌روزرسانی‌های بعدی همین پوشه:

```powershell
git pull --ff-only origin v7
python -m pip install -r requirements_v7.txt
python verify_v7.py
```

داده را می‌توانید از پوشهٔ قبلی مصرف کنید. لازم نیست دوباره کپی شود:

```powershell
$DataRoot = './datasets'
```

| دیتاست | فایل raw مورد انتظار در DataRoot |
|---|---|
| synthetic | خود برنامه با seed تولید می‌کند |
| ml-100k | `ml-100k/u.data` |
| ml-1m | `ml-1m/ratings.dat` |
| ml-10m | `ml-10M100K/ratings.dat` یا `ml-10m/ratings.dat` |
| jester | فایل‌های `jester/jester-data-*.xls`؛ یا CSV سازگار با loader |
| bookcrossing | `bookcrossing/BX-Book-Ratings.csv` |

دفترچهٔ دریافت داده در `vendor_v8/download_datasets_q1_v8.ipynb` موجود است. برای اجرا، مسیر DataRoot باید همان مسیر raw چاپ‌شده/آماده‌شده باشد. plan، دیتاست را دانلود نمی‌کند. قبل از مطالعهٔ `all` دادهٔ همهٔ دیتاست‌های واقعی را آماده کنید.

## مفهوم config و هایپرپارامتر

config داخل `examples/v7/common/ml-100k.json` نقطهٔ شروع، پروتکل و بودجهٔ جست‌وجو را تعیین می‌کند؛ به معنی بهینه‌بودن آن نیست. ظرفیت بیشتر نیز تضمین دقت نیست. برای هر شش دیتاست، `common`، `development` و `paper` فایل مستقل دارند.

- `common`: تقسیم per-user مشابه پایلوت V6، حدود ۸۰/۱۰/۱۰؛ مناسب مقایسهٔ توسعهٔ نسخه‌ها. نقش‌های validation برای انتخاب و probability fit/holdout جدا هستند.
- `development` و `paper` روی دادهٔ واقعی: پروتکل global مقالهٔ AEMC، ۸۵٫۵/۴٫۵/۱۰؛ همهٔ validation برای انتخاب، Method7 به‌صورت point-only. تغییر پروتکل، TEST را تغییر می‌دهد.
- `search/best_configurations.json`: خروجی انتخابِ هر مدل و outer split؛ فایلِ قالب ورودی `--v7-config` نیست. `final` خودش آن را همراه منشأ انتخاب مصرف می‌کند.

پارامتر و epoch با validation انتخاب می‌شوند. TEST برای انتخاب استفاده نمی‌شود. `main` فقط config ثابت را اجرا می‌کند؛ `develop` جست‌وجو و سپس `final` و گزارش منتخب‌ها را انجام می‌دهد. در `paper_v7.py`، مرحلهٔ `scan` فقط جست‌وجوست. در فرمان مستقیم `run_v7.py scan` برای توقف پیش از TEST باید `--scan-only` بدهید.

## سطح اول: سالم‌بودن اجرا و پایلوت ثابت

```powershell
python run_v7.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v7/smoke --v7-device cuda --v7-config ./examples/v7/common/ml-100k.json --quick --no-epoch-train-rmse --resume
```

این سه epoch برای صحت اجراست. برای عدد دقت از آن استفاده نکنید. پایلوت ثابتِ محدود:

```powershell
python run_v7.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v7/fixed_light --v7-config ./examples/v7/common/ml-100k.json --v7-device cuda --v7-max-epochs 120 --v7-steps-per-epoch 2 --v7-baseline-max-epochs 200 --v7-host-budget-mb 512 --v7-ssd-budget-gb 8 --no-epoch-train-rmse --resume
```

این فرمان بهترین hyperparameter را پیدا نمی‌کند. برای مقایسهٔ نسخه‌ها، seed، split، K، بودجه و پروتکل را گزارش کنید.

## سطح دوم: توسعهٔ تنظیم‌شدهٔ سبک

آرگومان‌ها را یک بار در همین جلسهٔ PowerShell تعریف کنید:

```powershell
$V7Light = @(
  '--profile','common','--datasets','ml-100k','--data-root',$DataRoot,
  '--output','./results_v7/tuned_light','--device','cuda',
  '--split-seeds','2026','--model-seeds','1','--k','20',
  '--search-jobs-per-method','6','--max-epochs','120',
  '--steps-per-epoch','2','--baseline-max-epochs','200',
  '--host-budget-mb','512','--ssd-budget-gb','8'
)
python paper_v7.py --stage plan @V7Light
python paper_v7.py --stage develop @V7Light
```

برنامهٔ این تنظیم ۱۰۷ آموزش بیرونیِ اسکن و ۱۵ آموزش نهایی دارد؛ Method2 داخل هر کاندیدا OOF expert هم آموزش می‌دهد. چند مقدار برای هر coefficient الزاماً در شش job جا نمی‌شوند؛ جدول coverage مشخص می‌کند چه چیز واقعاً تغییر کرده است. سقف هشت، فضای بیشتری برای ضرایب loss دارد. سقف صفر گرید بدون سقف است و ممکن است بسیار سنگین باشد.

خروجی همین develop شامل جدول TEST منتخب‌ها، Excel، پیش‌بینی‌ها و شکل‌هاست. نیازی به اجرای `main` پس از آن نیست. برای قطع و ادامه، همان فرمان را تکرار کنید؛ recipe خودش `--resume` دارد و به افزودن این گزینه به `paper_v7.py` نیاز نیست.

برای تشخیص حساسیت ضرایب، بدون تغییر برنده‌ها:

```powershell
python paper_v7.py --stage lossstudy @V7Light
```

اگر تحلیل حساسیت را برای تصمیم به گسترش فضای جست‌وجو مصرف می‌کنید، آن را پیش از دیدن TEST انجام دهید: `scan`، سپس `lossstudy`، سپس مطالعهٔ جست‌وجوی تازه و بعد `final`. lossstudy خودکار فایل برنده را عوض نمی‌کند. ضرایب bias/code/direction، reconstruction/auxiliary/consistency، precision/noise، transport، ensemble و probability در حد فعال‌بودن سازوکارشان ثبت/بررسی می‌شوند. بودجهٔ shrinkage و gradient surgery به‌عنوان نوع جدا گزارش می‌شوند.

## سطح سوم: مقاله روی ML-100K

پروفایل کامل زیر ۲۰ split، سه model seed و Kهای ۱۰/۲۰/۵۰ دارد. از ابتدا یک پوشه و یک بودجه ثابت انتخاب کنید:

```powershell
$V7Paper = @(
  '--profile','paper','--datasets','ml-100k','--data-root',$DataRoot,
  '--output','./results_v7/paper_ml100k','--device','cuda',
  '--host-budget-mb','512','--ssd-budget-gb','8'
)
python paper_v7.py --stage plan @V7Paper
python paper_v7.py --stage scan @V7Paper
python paper_v7.py --stage lossstudy @V7Paper
python paper_v7.py --stage final @V7Paper
python paper_v7.py --stage ablation @V7Paper
python paper_v7.py --stage ablation-fixed @V7Paper
python paper_v7.py --stage kstudy @V7Paper
python paper_v7.py --stage sparsity @V7Paper
python paper_v7.py --stage convergence @V7Paper
python paper_v7.py --stage temporal @V7Paper
python paper_v7.py --stage report @V7Paper
python paper_v7.py --stage audit @V7Paper
```

`ablation` از config منتخب هر مدل/split شروع می‌کند و یک مؤلفه را حذف می‌کند؛ انتخاب مجدد با TEST ندارد. `ablation-fixed` کنترل K ثابت است. `kstudy` حساسیت به K و `sparsity` چهار نرخ نگهداری TRAIN را بررسی می‌کنند. `temporal` تقسیم زمانیِ مستقل، با اسکن و انتخاب خودش است. `convergence` منحنی TRAIN/VAL را با بودجهٔ منتخب ثبت می‌کند. `report` و `audit` آموزش تازه ندارند.

معادل خودکار کل مجموعه:

```powershell
python paper_v7.py --stage all @V7Paper --include-temporal
```

all خروجی sensitivity را به‌عنوان تحلیل توصیفی نگه می‌دارد و config انتخاب‌شده را بی‌صدا تغییر نمی‌دهد. اگر فعلاً بودجهٔ کوچک‌تری برای مطالعهٔ مقاله می‌خواهید، مثلاً `--split-seeds 4101,4102,4103,4104,4105 --model-seeds 11,22 --k 10,20,50 --search-jobs-per-method 6` را به تعریف V7Paper اضافه و پوشهٔ تازه انتخاب کنید. این نتیجه، مطالعهٔ ۲۰splitی پیش‌فرض نیست.

تحلیل probability روی common و با نقش‌های disjoint انجام می‌شود:

```powershell
python paper_v7.py --stage calibration @V7Light
```

در paper/aemc-paper، Method7 point-only است؛ اجرای calibration روی آن نقش‌های احتمالاتیِ مستقل ایجاد نمی‌کند. برای synthetic، density نیز در all اجرا می‌شود. برای همهٔ دیتاست‌ها:

```powershell
python paper_v7.py --stage plan --profile paper --datasets all --data-root $DataRoot --output ./results_v7/paper_all
python paper_v7.py --stage all --profile paper --datasets all --data-root $DataRoot --output ./results_v7/paper_all --device cuda --host-budget-mb 512 --ssd-budget-gb 8 --include-temporal
```

زمان همهٔ شش دیتاست، جمع زمان ML-100K ضربدر شش نیست. ML-10M/Jester/BookCrossing بار متفاوت دارند و به پایلوت خودشان نیاز دارند.

## منابع، resume و seed

- یک GPU، یک job در هر لحظه، فرایند جدا برای jobها؛ مدل و cache قبلی قبل از بعدی آزاد می‌شوند. cache tensor داده فقط یک split دارد.
- backward تاریخچهٔ gated ماتریس چگال U×I نمی‌سازد. snapshot اولیهٔ مشترکِ replay نیز در حد بودجهٔ RAM، و در غیر آن روی SSD نگه داشته می‌شود.
- RAM/SSD محل ذخیره را تغییر می‌دهند؛ dtype، loss، batch منطقی و تعداد update خودکار عوض نمی‌شوند. اختلاف گردکردن CPU/CUDA ممکن است؛ CUDA واقعیِ تازه در محیط توسعه موجود نبود.
- host-budget سقف کل RAM نیست. ssd-budget سهمیهٔ موقت است و سقف نتایج مقاله نیست. امتیاز و config و historyِ همهٔ trialها نگه داشته می‌شود؛ آرشیوهای حجیمِ اسکن به رهبران محدود می‌شوند.
- همان کد، داده، config، seed، بودجه و output را برای resume نگه دارید. پس از ارتقای علمیِ کد، پوشهٔ تازه لازم است. job کامل retrain نمی‌شود؛ job جزئی از کاندیدای/epoch معتبر ادامه می‌یابد.
- Method8 برای حفظ trainer اصلی، داخل epoch بازیابی optimizer ندارد؛ کاندیدای کامل reuse و کاندیدای نیمه‌تمام از trainer اصلی شروع می‌شود.
- `--manual-seed 17` جای model-seeds و search seed را مشخص می‌کند؛ split-seeds مستقل است. آن را همزمان با `--model-seeds` ندهید. `--deterministic` کنترل تکمیلی است و برابری بین همهٔ سخت‌افزارها/نسخه‌های کتابخانه را تضمین نمی‌کند.

## زمان بسیار تقریبی روی 4090 با RAM شانزده گیگابایت

این‌ها اندازه‌گیری روی 4090 نیستند. مرجع قبلی پایلوت 5090 است؛ early stopping، ظرفیت، SSD و fallback می‌تواند زمان را چند برابر تغییر دهد.

| مرحله | common تک‌split، تخمین کلی | paper کامل ML-100K، تخمین کلی |
|---|---|---|
| plan | چند ثانیه تا ۲ دقیقه | چند ثانیه تا ۲ دقیقه |
| smoke | ۱–۵ دقیقه | برای گزارش دقت نیست |
| main | ۱۰–۶۰ دقیقه؛ light اغلب کمتر | ۲۱–۱۶۹ ساعت |
| scan | حدود ۰٫۵–۶ ساعت | ۱۹۵–۱۵۶۰ ساعت |
| final | ۵–۴۵ دقیقه | ۴–۳۴ ساعت |
| lossstudy | حدود ۱–۱۲ ساعت | ۸۱–۶۴۶ ساعت |
| ablation، هر حالت | حدود ۰٫۵–۶ ساعت | ۳۷–۲۹۸ ساعت |
| kstudy | با چند K حدود ۰٫۲–۲ ساعت | ۱۳–۱۰۱ ساعت |
| sparsity | حدود ۰٫۳–۳ ساعت | ۱۷–۱۳۷ ساعت |
| convergence | حدود ۵–۶۰ دقیقه | ۴–۳۴ ساعت |
| calibration در common | چند دقیقه تا حدود نیم ساعت | در paper point-only اجرا نمی‌شود |
| temporal scan+final | تقریباً یک scan+final دیگر | حدود ۱۹۹–۱۵۹۴ ساعت |
| report / audit | چند ثانیه تا چند ده دقیقه، عمدتاً CPU/SSD | وابسته به حجم نتایج، نه آموزش GPU |
| all بدون temporal | حدود ۴–۳۰ ساعت | حدود ۳۸۹–۳۱۰۹ ساعت |

جدول common از برنامهٔ پیش‌فرضِ بدون cap کوتاه می‌آید؛ زمان light با capهای معرفی‌شده معمولاً کمتر است، ولی کاهش خطی زمان به نسبت epoch فرض نمی‌کنیم. تعدادها با معماری منتخب تغییر می‌کنند. برای همان فرمان خودتان برنامه را بشمارید و بعد از یک پایلوتِ همان دستگاه، CSV زمان واقعی را بدهید:

```powershell
python estimate_v7.py --profile common --search-jobs-per-method 6 --max-epochs 120 --steps-per-epoch 2 --baseline-max-epochs 200
python estimate_v7.py --profile paper
python estimate_v7.py --profile common --calibration ./results_v7/fixed_light/ml-100k/main/candidate_validation_runs.csv
```

## فایل‌هایی که باید گزارش کنید

`search/best_configurations.json`، `final_selection_provenance.json`، `final_locked_test_summary.csv`، `publication/research_tables.xlsx`، `search_parameter_coverage.csv`، جدول‌های مستقل ablation/lossstudy و `figures/` مرجع‌اند. seedها ابتدا داخل هر split میانگین و سپس بین splitها جمع‌بندی می‌شوند. یک split، CI مستقل ایجاد نمی‌کند. SVG/PDF برداری و PNG با 400 DPI تولید می‌شود؛ CSV و Excel برای رسم دوباره حفظ می‌شوند.

```powershell
python run_v7.py report --datasets ml-100k --output ./results_v7/tuned_light
python run_v7.py curves --datasets ml-100k --output ./results_v7/tuned_light
```

این دو فرمان آموزش جدید نیستند. برای نمودار دلخواه از Excel، ابزار `replot_v7.py` و مثال بخش ۹ راهنمای جامع را به کار ببرید.
