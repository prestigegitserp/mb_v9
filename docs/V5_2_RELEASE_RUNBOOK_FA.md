# دریافت نسخهٔ پنجم، اجرای V5.2 و ارتباط آزمایش‌ها

سورس V5.2 با اجازهٔ جدید کاربر کامیت و در `v5-development` ادغام و روی `v5` منتشر شده است. برای شناسهٔ نهایی commit به خود شاخه‌های GitHub رجوع کنید. نسخه‌های ۱ تا ۴ نباید جابه‌جا یا بازنویسی شوند. نخستین commit این انتشار، سورس دقیق tree قبلی `016df6ac193f2f296813617c7fd21bbbf2226e05` را محفوظ کرده است؛ رفع اشکال اجرای واقعی و این راهنما در commit بعدی مشخص‌اند.

## ۱. پوشهٔ خالی در Windows / PowerShell

ترمینال VS Code را داخل پوشهٔ خالیِ انتخابی باز کنید:

```powershell
git clone --config core.autocrlf=false --branch v5 --single-branch https://github.com/prestigegitserp/mb_v9.git .
git branch --show-current
git log -1 --oneline
conda activate foo311
python -m pip install -r requirements_v5_2.txt
python verify_v5_2.py
python -c "import sys,torch,v52; print(sys.executable); print(v52.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

گزینهٔ core.autocrlf=false فقط برای همین مخزن ذخیره می‌شود و پایان‌خط فایل‌های آرشیوی را در Windows حفظ می‌کند تا بررسی SHA فایل‌هایی مثل SVG و PowerShell نیز عین سورس باشد. برای نخستین دریافت در پوشهٔ خالی از `clone` استفاده می‌شود. پس از آن، دریافت به‌روزرسانی‌های همین شاخه:

```powershell
git pull --ff-only origin v5
```

اگر خود شاخهٔ توسعه را می‌خواهید، نام `v5` را در هر دو فرمان بالا به `v5-development` تغییر دهید. `main` نسخهٔ پنجم نیست. برای این انتشار از `paper_v5_2.py` یا `run_v5_2.py` استفاده کنید؛ فایل قدیمی `run_v9.py` دست‌نخورده و ورودی V5 اولیه است.

در VS Code نیز interpreter محیط `foo311` را انتخاب کنید. خروجی بررسی CUDA باید `True` باشد تا `--device cuda:0` کار کند. نصب requirements به‌تنهایی جایگزینی اجباری PyTorch فعال CUDA با نسخهٔ CPU نیست.

## ۲. بررسی سلامت و smoke

```powershell
python -m pytest tests/test_v52_models.py tests/test_v52_goal.py tests/test_v52_workflow.py -q -ra
python -m pytest tests -q -ra
python paper_v5_2.py --phase smoke --dataset ml-100k --data-root ./datasets --output ./paper_v52 --device cuda:0
```

smoke به `paper_v52/smoke/ml-100k` می‌رود، بودجهٔ بسیار کوچک و انتخاب محدود دارد و نتیجهٔ مقاله محسوب نمی‌شود. این مرحله مسیر داده، مدل، دستگاه و گزارش را بررسی می‌کند. باقی آزمایش‌های اصلی در `paper_v52/ml-100k` قرار می‌گیرند.

در بررسی پیش از انتشار، تست‌های اختصاصی ۳۷ موفق و مجموعهٔ کامل ۳۵۰ موفق / ۳ skipped داشته‌اند. این‌ها تست CPU و fixture مصنوعی‌اند؛ آموزش اصلی MovieLens، اثبات RMSE زیر0.87 و آزمون CUDA واقعی نیستند. جزئیات و نسخهٔ محیط در `docs/v52_evidence/RUNTIME_VALIDATION.json` ثبت می‌شوند.

## ۳. تنظیمات مشترک آزمایش اصلی

این آرایه را یک بار در PowerShell تعریف کنید؛ سپس همهٔ مراحل همان تنظیمات را می‌گیرند:

```powershell
$paperArgs = @(
  '--dataset', 'ml-100k',
  '--data-root', './datasets',
  '--output', './paper_v52',
  '--protocol', 'aemc-paper',
  '--config', './examples/v52/high_capacity_ml100k.json',
  '--split-seeds', ((4101..4120) -join ','),
  '--model-seeds', '11,22,33',
  '--k', '20,50,100',
  '--scan-mode', 'hybrid',
  '--device', 'cuda:0',
  '--storage', 'auto',
  '--host-budget-mb', '1024',
  '--ssd-budget-gb', '64',
  '--cache-dir', './v52_ssd_cache'
)
python paper_v5_2.py --phase plan @paperArgs
```

این طرح واقعاً سنگین است. بررسی `plan` با این پیش‌فرض‌ها ۱۴ خانواده، ۴۱۶ scan job و **۶۹٬۷۸۰ آموزش کاندیدا فقط در scan** گزارش کرده است. تعداد مدل‌های هر job، seedها، Kها و initهای MBMF داخل این عددند. زمان واقعی روی لپ‌تاپ اندازه‌گیری نشده است. اعداد برنامهٔ ابلیشن، برآورد پیش از مشخص‌شدن winnerها هستند؛ تعداد دقیق اجرای پس از انتخاب و هزینهٔ واقعی نیز ثبت می‌شوند.

برای بررسی اولیه روی لپ‌تاپ، مطالعهٔ محدود و جدا بسازید:

```powershell
python paper_v5_2.py --phase plan --dataset ml-100k --data-root ./datasets --output ./v52_development --split-seeds 4101 --model-seeds 11 --k 20 --search-jobs-per-method 4 --max-epochs 80 --baseline-max-epochs 300
python paper_v5_2.py --phase scan --dataset ml-100k --data-root ./datasets --output ./v52_development --split-seeds 4101 --model-seeds 11 --k 20 --search-jobs-per-method 4 --max-epochs 80 --baseline-max-epochs 300 --device cuda:0
```

این مطالعهٔ محدود، grid کامل یا بازتولید مقاله نیست. تغییر grid، seed، K، budget یا source پس از scan، مطالعهٔ تازه و پوشهٔ خروجی تازه می‌خواهد. برای تصمیم‌های توسعه از validation استفاده کنید؛ TEST آزمایش تأییدی را برای اصلاح مجدد مدل مصرف نکنید.

## ۴. ترتیب اصلی و ارتباط نتایج

| مرحله | چه چیزی می‌آموزد یا می‌خواند؟ | نقش نتیجه |
|---|---|---|
| `plan` | config و تعداد jobها؛ بدون آموزش | برآورد دامنهٔ کار |
| `scan` | آموزش روی TRAIN؛ انتخاب با validation | winner مستقل هر خانواده در هر outer split |
| `final` | config قفل‌شدهٔ scan؛ refit با seedهای ثابت؛ TEST نهایی | جدول اصلی مقایسهٔ مقاله |
| `ablation` | تنظیمات winner؛ حذف اجزای مدل و آموزش مجدد | سهم هر مکانیسم با K منتخب همان split |
| `ablation-fixed` | تنظیمات winner؛ K مشترک۲۰ برای full و حذف‌ها | کنترل اثر بُعد نهفته در ابلیشن |
| `kstudy` | تنظیمات winner؛ تغییر K | حساسیت بُعد، بدون انتخاب بهترین ردیف TEST |
| `sparsity` | تنظیمات winner؛ کاهش فقط یال‌های TRAIN | پایداری با دادهٔ کمتر؛ val/test ثابت |
| `convergence` | تنظیمات winner؛ TRAIN/validation در tickهای ثبت | رفتار یادگیری؛ TEST هر epoch اندازه‌گیری نمی‌شود |
| `report` | فایل‌های ذخیره‌شده | بازتولید جداول، Excel و شکل‌ها؛ بدون آموزش |
| `audit` | provenance انتخاب و مرز داده‌ها | بررسی انتخاب فقط با validation |
| `assess` | آرشیو TEST و config نهایی | حسابرسی هدف0.87 و مقایسهٔ همهٔ مدل‌ها |

### اسکن و آزمایش اصلی

```powershell
python paper_v5_2.py --phase scan @paperArgs
python paper_v5_2.py --phase final @paperArgs
```

launcher در `scan` صریحاً `--scan-only` را فعال می‌کند: TRAIN/validation، نمودارهای قابل تولید در این مرحله، تاریخچه، جداول انتخاب و workbook تولید می‌شوند. فایل‌های TEST متعلق به `final` هستند. دستور مستقیم `run_v5_2.py scan` بدون `--scan-only` رفتار متفاوتی دارد و final را هم ادامه می‌دهد؛ برای روند مرحله‌ای از launcher بالا استفاده کنید.

فایل‌های اصلی انتخاب:

- `paper_v52/ml-100k/search/hyperparameter_validation_all.csv`
- `paper_v52/ml-100k/search/validation_winners.csv`
- `paper_v52/ml-100k/search/validation_selection_manifest.json`
- `paper_v52/ml-100k/search/publication/research_tables.xlsx`

`final` همان تنظیم منتخب، K، optimizer و checkpoint policy را می‌گیرد؛ تنظیمات baseline و عضویت TEST نیز کنترل می‌شوند. validation به TRAIN اضافه نمی‌شود. خروجی اصلی:

- `paper_v52/ml-100k/final_locked_test_all.csv`
- `paper_v52/ml-100k/final_selection_provenance.json`
- `paper_v52/ml-100k/publication/research_tables.xlsx`
- `paper_v52/ml-100k/final/<method>/split_<seed>/` برای checkpoint، prediction و گزارش هر مدل.

پروتکل `aemc-paper` روی ML-100K همان نسبت85.5/4.5/10 را به کار می‌برد. هر خانواده همان4500 زوج selection دارد. Method7 در این مطالعه point-only است؛ probability fitting اضافه انجام نمی‌دهد. تطابق تنظیمات، به‌تنهایی بازتولید دقت مقاله را ثابت نمی‌کند.

### ابلیشن، حساسیت، دادهٔ کم و همگرایی

```powershell
python paper_v5_2.py --phase ablation @paperArgs
python paper_v5_2.py --phase ablation-fixed @paperArgs
python paper_v5_2.py --phase kstudy @paperArgs
python paper_v5_2.py --phase sparsity @paperArgs
python paper_v5_2.py --phase convergence @paperArgs
```

همهٔ این مراحل به‌طور پیش‌فرض `--analysis-source winners` دارند؛ از frozen winner اسکن شروع می‌کنند. در ابلیشن، حذف جزء و آموزش مجدد رخ می‌دهد و بودجهٔ واقعی گزارش می‌شود. در kstudy، منحنی TEST برای حساسیت توصیفی است و جایگزین انتخاب K با validation نمی‌شود. sparsity پیش‌فرض30/50/70/100 درصد TRAIN را با حفظ پشتیبانی user/item بررسی می‌کند؛ نسبت تحقق‌یافته نیز ثبت می‌شود.

خروجی‌ها به‌ترتیب در `ablations/`، `ablations_matched_K_20/`، `kstudy/`، `sparsity/` و `convergence/` زیر پوشهٔ دیتاست هستند. همگرایی با فاصلهٔ checkpoint همان winner ثبت می‌شود؛ پیش‌فرض معمولاً هر۵ epoch است، نه الزاماً تک‌تک epochها. اگر ثبت هر epoch می‌خواهید، از ابتدا در ورودی مستقیم `run_v5_2.py`، گزینهٔ `--eval-every 1` را برای کل مطالعه و اسکن/ارزیابی یکسان تعیین کنید؛ تغییر آن می‌تواند تعداد tickهای patience و آموزش را هم تغییر دهد.

### اجرای همهٔ مراحل و ادامه پس از قطع

```powershell
python paper_v5_2.py --phase all @paperArgs
```

`all` شامل scan، final، ابلیشن‌ها، K، sparsity و convergence است و مرحله‌های کاملِ مطابق config را resume می‌کند. در پروتکل point-only، calibration صریحاً skip می‌شود. `all` به‌صورت پیش‌فرض temporal را اجرا نمی‌کند. با همان آرگومان‌ها، پس از قطع دوباره همین فرمان را اجرا کنید؛ config متفاوت در پوشهٔ قدیمی بی‌سروصدا پذیرفته نمی‌شود. `assess` و بازتولید نهایی گزارش را به‌صورت صریح در مرحلهٔ بعد اجرا کنید.

## ۵. گزارش نهایی، کنترل انتخاب و هدف0.87

```powershell
python paper_v5_2.py --phase report @paperArgs
python paper_v5_2.py --phase audit @paperArgs
python paper_v5_2.py --phase assess @paperArgs
```

این مراحل آموزش تازه ندارند. report فایل‌های CSV، Excel و شکل‌های PNG/PDF/SVG را از خروجی موجود می‌سازد. نبود prediction، history یا دادهٔ probability با دادهٔ ساختگی جبران نمی‌شود. نمودارهای احتمال در مطالعهٔ point-only موضوعیت ندارند.

assess از NPZ واقعی TEST، RMSE را دوباره حساب و زوج‌ها و truth را بین همهٔ روش‌ها تطبیق می‌دهد. جمعیت85500/4500/10000، 943کاربر/1682آیتم، config قفل‌شده و source hash نیز بررسی می‌شوند. میانگین model seedها ابتدا داخل هر split و سپس میانگین outer splitها محاسبه می‌شود. هر هفت روش گزارش می‌شوند؛ بهترین ردیف TEST انتخاب و به اسکن بازگردانده نمی‌شود.

خروجی در `paper_v52/ml-100k/target_087/` شامل `goal_by_method.csv`، `paired_deltas.csv`، `archive_integrity.csv`، `protocol_checks.csv` و `GOAL_ASSESSMENT.json` است. زیر0.87، مقایسه با هر هفت رقیب و وضعیت مرجع AEMC جدا گزارش می‌شوند. آستانه0.89 برای پرچم مرجع ضعیف صرفاً tolerance تشخیصی است. کنترل decoder-only شواهد برتری هندسی شمرده نمی‌شود؛ عبور عددی به‌تنهایی نوآوری یا معنی‌داری آماری را ثابت نمی‌کند.

## ۶. شکل تازه از Excel یا CSV، بدون آموزش

```powershell
python replot_v5_2.py --input ./paper_v52/ml-100k/publication/research_tables.xlsx --sheet metrics_long --where metric=test_RMSE --x method --y value --kind barh --aggregate mean --output ./paper_figures_v52/rmse.png
python replot_v5_2.py --input ./paper_v52/ml-100k/publication/tables/metrics_long.csv --where metric=test_MAE --x method --y value --kind box --output ./paper_figures_v52/mae_distribution.png
```

بازترسیم، PNG/PDF/SVG و CSV/JSON مربوط به خود شکل را نگه می‌دارد. داده‌های per-pair کامل در `publication/tables/per_pair/*.csv.gz` و predictionهای NPZ هستند؛ به محدودیت ردیف Excel بریده نمی‌شوند. workbook، INDEX، SHA و ارجاع به CSVهای کامل دارد. برای تنظیم شکل دلخواه، `python replot_v5_2.py --help` را ببینید.

## ۷. آزمایش احتمالات Method7؛ مطالعهٔ جدا

```powershell
python paper_v5_2.py --phase scan --dataset ml-100k --protocol common --methods Method7 --output ./paper_v52_probability --device cuda:0
python paper_v5_2.py --phase calibration --dataset ml-100k --protocol common --methods Method7 --output ./paper_v52_probability --device cuda:0
python paper_v5_2.py --phase report --dataset ml-100k --protocol common --methods Method7 --output ./paper_v52_probability
```

این پروتکل selection، probability-fit و probability-hold را جدا می‌کند. calibration با معیار احتمال validation انتخاب و جدا گزارش می‌شود؛ نتیجهٔ آن همان مطالعهٔ point-only هدف0.87 نیست. اگر جدول RMSE نهایی همین مطالعه را هم می‌خواهید، phase `final` را با همین آرگومان‌ها اجرا کنید.

## ۸. آزمایش‌های تکمیلیِ مستقل

ورودی مستقیم برای گزینه‌هایی است که launcher کوتاه آن‌ها را نشان نمی‌دهد. در هر مطالعه output مستقل داشته باشید.

```powershell
python run_v5_2.py temporal --datasets ml-100k --data-root ./datasets --output ./paper_v52_temporal --split-seeds 4101 --model-seeds 11 --k 20 --v52-protocol common --v52-device cuda:0 --v52-search-jobs-per-method 4 --resume
python run_v5_2.py density --datasets synthetic --data-root ./datasets --output ./paper_v52_density --split-seeds 4101 --model-seeds 11 --k 20 --density-rates 0.05,0.10,0.20,0.40,0.60 --v52-protocol common --v52-device cuda:0 --resume
```

temporal اسکن و final خودش را روی تقسیم زمانی انجام می‌دهد؛ با تقسیم تصادفی مقاله یکی نیست. density، تراکمِ دادهٔ مصنوعیِ تولیدشده را تغییر می‌دهد و جمعیت held-out ممکن است عوض شود؛ با thinning ثابتِ TRAIN در sparsity تفاوت دارد. این فرمان‌های تکمیلی با بودجهٔ محدود، بررسی اولیه‌اند.

برای دیتاست‌های دیگر، همین launcher را با `--dataset ml-1m`، `ml-10m`، `jester`، `bookcrossing` یا `synthetic` و output مستقل اجرا کنید. خانواده‌های پشتیبانی‌شده به دیتاست وابسته‌اند. پیش‌فرض MovieLens، `aemc-paper` و پیش‌فرض سایر داده‌ها `common` است؛ preset پهن ML-100K فقط به همان داده به‌صورت خودکار داده می‌شود. ارزیاب `assess` هدف0.87 مخصوص ML-100K است.

## ۹. GPU، RAM و SSD

`--device cuda:0` یک GPU را درخواست می‌کند؛ `auto` در نبود CUDA به CPU می‌رود. storage خودکار ابتدا حافظهٔ GPU، سپس بودجهٔ host و SSD را مدیریت می‌کند؛ از دست رفتن حافظه یا شکست عددیِ قابل بازیابی می‌تواند candidate را از حالت اولیه روی CPU بازاجرا کند. دستگاه واقعی، علت fallback و مصرف منابع ثبت می‌شوند. تغییر dtype، batch منطقی یا objective به‌عنوان ترفند حافظه انجام نمی‌شود؛ برابری بیتی CPU/CUDA وعده داده نمی‌شود. با تغییر فقط محل cache، هدف علمی تغییر نمی‌کند.

گزینه‌های `--storage auto`، `--host-budget-mb`، `--ssd-budget-gb` و `--cache-dir` را با ظرفیت واقعی سیستم تنظیم کنید. تست CPU موفق، تضمین نهایت سرعت یا آزمون واقعی RTX3050 نیست؛ وضعیت CUDA و هزینهٔ واقعی باید روی همان لپ‌تاپ ثبت شود.
