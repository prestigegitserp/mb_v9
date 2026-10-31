# انتشار ۸.۰.۰: معادل کامل راهنمای انتشار نسخه۷، با مراحل روشن TEST

شاخهٔ انتشار `v8` و شاخهٔ توسعه `v8_development` است. این راهنما، معادل اجرایی [راهنمای انتشار نسخه۷](V7_RELEASE_RUNBOOK_FA.md) را پوشش می‌دهد و مکمل [راهنمای جامع فعلی نسخه۸](V8_RUNBOOK_FA.md) است. مطالب راهنمای جامع، تحلیل مدل‌ها و شواهد قبلی حفظ شده‌اند. برای معنی تک‌تک مراحل و گزینه‌ها، [مرجع فرمان‌ها](V8_COMMAND_REFERENCE_FA.md) را ببینید.

هر هشت نام عمومی Method1 تا Method8 و baselineهای مجاز دیتاست حفظ شده‌اند. فایل‌های نسخه‌های۱تا۷، VERSION ریشه، منبع اصلی Method10 و ورودی‌های قدیمی تغییر نکرده‌اند. `VERSION_V8` و `v8.__version__` نسخهٔ انتشار جدید را مشخص می‌کنند.

## ۱. دریافت در پوشهٔ خالی و به‌روزرسانی

داخل پوشهٔ واقعاً خالی، در PowerShell یا ترمینال Linux:

```text
git clone --branch v8 --single-branch https://github.com/prestigegitserp/mb_v9.git .
python -m pip install -r requirements_v8.txt
python verify_v8.py
python -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

بهتر است محیط Python مستقلی داشته باشید. `--device cuda` فقط وقتی قابل استفاده است که PyTorch نصب‌شده CUDA را ببیند. `--device auto` دستگاه موجود را انتخاب می‌کند و دستگاه واقعی در گزارش ثبت می‌شود.

برای به‌روزرسانی همین checkout انتشار، پس از پایان اجراهای فعال:

```text
git pull --ff-only origin v8
python -m pip install -r requirements_v8.txt
python verify_v8.py
```

اگر قبلاً شاخهٔ توسعه را با `--single-branch` گرفته‌اید و می‌خواهید نسخهٔ انتشار را در همان checkout داشته باشید، ابتدا وضعیت فایل‌های خودتان را با `git status` بررسی کنید؛ سپس:

```text
git fetch origin v8:refs/remotes/origin/v8
git switch --track origin/v8
```

فرمان دوم برای اولین ساخت شاخهٔ محلی v8 است؛ اگر از قبل ساخته شده، `git switch v8` کافی است. گزینهٔ دیگر، clone جدا در پوشهٔ تازه است. برای توسعه، به‌جای v8 شاخهٔ `v8_development` را clone کنید. مطالعهٔ شروع‌شده با نسخهٔ توسعهٔ قبلی را با سورس انتشار جدید مخلوط نکنید؛ برای انتشار یک `--output` تازه بدهید.

## ۲. DataRoot و شش دیتاست

می‌توانید دادهٔ نسخه۷ را مستقیم مصرف کنید. در PowerShell:

```powershell
$DataRoot = './datasets'
```

| دیتاست | فایل raw مورد انتظار زیر DataRoot | نکته |
|---|---|---|
| synthetic | ساخته‌شده توسط برنامه | دانلود لازم ندارد |
| ml-100k | `ml-100k/u.data` | مناسب اولین بررسی واقعی |
| ml-1m | `ml-1m/ratings.dat` | نیازمند پایلوت مستقل |
| ml-10m | `ml-10M100K/ratings.dat` یا `ml-10m/ratings.dat` | هزینه و حافظهٔ بیشتر |
| jester | `jester/jester-data-*.xls` یا CSV سازگار با loader | مقیاس rating با MovieLens متفاوت است |
| bookcrossing | `bookcrossing/BX-Book-Ratings.csv` | قواعد preprocessing قبلی حفظ شده‌اند |

دفترچهٔ دریافت داده: [download_datasets_q1_v8.ipynb](../vendor_v8/download_datasets_q1_v8.ipynb). `plan` داده را دانلود نمی‌کند. قبل از اجرای همهٔ دیتاست‌ها، rawهای واقعی را آماده کنید. در Linux و فرمان‌های یک‌خطی زیر، `./datasets` را با مسیر واقعی خودتان جایگزین کنید؛ برای مسیر دارای فاصله، کل مسیر را داخل double quote بگذارید.

## ۳. config، پروتکل و معنی «بهینه»

`examples/v8/<profile>/<dataset>.json` نقطهٔ شروع مطالعه است. سه پروفایل برای هر شش دیتاست وجود دارند:

| پروفایل | تقسیم روی دادهٔ واقعی | متد۷ | کاربرد |
|---|---|---|---|
| common | per-user حدود۸۰/۱۰/۱۰ | احتمال با نقش‌های مجزای VAL | مقایسه با پایلوت قبلی و پژوهش uncertainty |
| development | global حدود۸۵٫۵/۴٫۵/۱۰ | point-only | توسعه تحت پروتکل مقالهٔ AEMC |
| paper | همان پروتکل global، بودجه و تکرار بیشتر | point-only | مطالعهٔ بزرگ‌تر تحت آن پروتکل |

این نسبت‌ها برای synthetic قرارداد تولید دادهٔ خودش را جایگزین نمی‌کنند. تغییر پروفایل، جمعیت TEST را تغییر می‌دهد؛ اختلاف امتیاز دو پروتکل را به بهبود معماری نسبت ندهید.

`main` یا `pilot` config ثابت را اجرا می‌کند. در `scan`، پارامترها، K، initialization، checkpoint و postfit با VAL انتخاب می‌شوند. سپس `test`، که همان `final` است، تنظیم منتخب را بازآموزی و TEST را ارزیابی می‌کند. `develop` هر دو را پشت سر هم انجام می‌دهد.

نکتهٔ مهم فرمان مستقیم: `python run_v8.py scan` به‌طور پیش‌فرض بعد از scan وارد TEST هم می‌شود؛ برای توقف پیش از TEST، `--scan-only` بدهید. در `paper_v8.py --stage scan` این گزینه خودکار اعمال می‌شود.

برندهٔ search یعنی بهترین کاندیدای واقعاً بررسی‌شده روی VAL، نه بهینهٔ جهانی. `best_configurations.json` خروجی جست‌وجوست و قالب ورودی `--v8-config` نیست؛ مرحلهٔ test خودش منشأ انتخاب را مصرف می‌کند.

## ۴. سطح اول: smoke و پایلوت ثابت

smoke با سقف سه epoch و اولین split/seed/K است؛ پیش‌بینی TEST و شکل‌ها دارد، اما شکل‌ها watermark دارند و شاهد دقت مقاله نیستند:

```powershell
python paper_v8.py --stage smoke --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v8/smoke --device cuda
python paper_v8.py --stage status --output ./results_v8/smoke
```

معادل مستقیم فرمان smoke نسخه۷:

```powershell
python run_v8.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v8/smoke_direct --v8-device cuda --v8-config ./examples/v8/common/ml-100k.json --quick --no-epoch-train-rmse --resume
```

پایلوت ثابتِ سبک، بدون search هایپرپارامتر:

```powershell
python paper_v8.py --stage pilot --profile common --budget light --datasets ml-100k --data-root $DataRoot --output ./results_v8/fixed_light --device cuda
```

معادل دقیق‌تر از نظر گزینه‌های صریحِ راهنمای نسخه۷:

```powershell
python run_v8.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v8/fixed_light_direct --v8-config ./examples/v8/common/ml-100k.json --v8-device cuda --v8-max-epochs 120 --v8-steps-per-epoch 2 --v8-baseline-max-epochs 200 --v8-host-budget-mb 512 --v8-ssd-budget-gb 8 --no-epoch-train-rmse --resume
```

دو مسیر بالا را در دو پوشهٔ جدا نشان داده‌ایم؛ اجرای هر دو برای یک مطالعه لازم نیست. در فرمان مستقیم، config و seed و بودجه را خودتان در ادامه ثابت نگه می‌دارید؛ مسیر paper تنظیمات را ذخیره می‌کند.

خروجی پایلوت در `ml-100k/main/` است: `locked_test_summary.csv`، `publication/research_tables.xlsx`، `publication/figures/selected_test_rmse.png` و آرشیوهای prediction. پایلوت ثابت برای انتخاب hyperparameter طراحی نشده است.

## ۵. سطح دوم: تنظیم‌شدهٔ سبک، معادل V7Light

این آرایه فقط syntax مربوط به PowerShell است؛ آن را داخل سلول Python یا Bash نگذارید:

```powershell
$V8Light = @(
  '--profile','common','--datasets','ml-100k','--data-root',$DataRoot,
  '--output','./results_v8/tuned_light','--device','cuda',
  '--split-seeds','2026','--model-seeds','1','--k','20',
  '--search-jobs-per-method','6','--max-epochs','120',
  '--steps-per-epoch','2','--baseline-max-epochs','200',
  '--host-budget-mb','512','--ssd-budget-gb','8'
)
python paper_v8.py --stage plan @V8Light
python paper_v8.py --stage develop @V8Light
```

این همان نقش اجرایی V7Light را دارد؛ تعداد و محتوای کاندیداهای نسخه۸ را خود plan می‌شمارد. عدد ثابت۱۰۷ نسخه۷ را بدون شمارش مجدد به نسخه۸ نسبت ندهید. در شمارش همین انتشار با این تنظیمات،۸۳ job و۱۰۷ آموزش کاندیدا، و۱۵ آموزش final تأیید شد؛ پایلوت ثابت۲۱ آموزش دارد. Method2 داخل یک کاندیدا expertهای OOF هم آموزش می‌دهد؛ تعداد آموزش بیرونی، کل کار داخلی نیست.

معادل یک‌خطیِ تعریف مطالعه، در PowerShell و Bash:

```text
python paper_v8.py --stage plan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/tuned_light --device cuda
```

بعد از این plan، هر دو مسیر زیر از recipe ذخیره‌شده استفاده می‌کنند. یکی را انتخاب کنید:

```text
python paper_v8.py --stage develop --output ./results_v8/tuned_light
```

یا، برای کنترل زمان مشاهدهٔ TEST:

```text
python paper_v8.py --stage scan --output ./results_v8/tuned_light
python paper_v8.py --stage lossstudy --output ./results_v8/tuned_light --loss-methods Method6 --loss-parameters transport_moment_weight
python paper_v8.py --stage test --output ./results_v8/tuned_light
python paper_v8.py --stage status --output ./results_v8/tuned_light
```

lossstudy فقط VAL است و winner را خودکار عوض نمی‌کند. اگر از حساسیت برای انتخاب فضای تازه استفاده می‌کنید، پیش از مشاهدهٔ TEST تصمیم بگیرید، مطالعهٔ تازه تعریف کنید و سپس TEST تأییدی بگیرید. اجرای همهٔ ضرایب با حذف دو فیلتر امکان‌پذیر است و می‌تواند طولانی باشد.

سقف۶ job برای هر روش، تمام ضرایب را کامل نمی‌گردد. کاندیدای نسخه۷، prior معتبر و مقادیر سازوکار در اولویت‌اند؛ پوشش واقعی را از `search/publication/tables/search_parameter_coverage.csv` بخوانید. `--search-jobs-per-method 8` دامنهٔ بزرگ‌تری می‌دهد؛ صفر سقف تعداد job را برمی‌دارد. این تغییرها را از ابتدا در خروجی تازه تعریف کنید.

## ۶. سطح سوم: مجموعهٔ مقاله روی ML-100K، معادل V7Paper

پیش‌فرض paper شامل۲۰ split، سه model seed و Kهای۱۰/۲۰/۵۰ است. این مطالعه بسیار بزرگ است. ابتدا plan و تخمین هزینه را ببینید؛ شروع paper به معنی کافی‌بودن خودکار شواهد برای ژورنال نیست.

```powershell
$V8Paper = @(
  '--profile','paper','--datasets','ml-100k','--data-root',$DataRoot,
  '--output','./results_v8/paper_ml100k','--device','cuda',
  '--host-budget-mb','512','--ssd-budget-gb','8'
)
python paper_v8.py --stage plan @V8Paper
python paper_v8.py --stage scan @V8Paper
python paper_v8.py --stage lossstudy @V8Paper
python paper_v8.py --stage test @V8Paper
python paper_v8.py --stage ablation @V8Paper
python paper_v8.py --stage ablation-fixed @V8Paper
python paper_v8.py --stage kstudy @V8Paper
python paper_v8.py --stage sparsity @V8Paper
python paper_v8.py --stage convergence @V8Paper
python paper_v8.py --stage temporal @V8Paper
python paper_v8.py --stage report @V8Paper --plots
python paper_v8.py --stage audit @V8Paper
python paper_v8.py --stage status @V8Paper
```

`--stage final` نیز دقیقاً معادل `--stage test` است. بعد از plan می‌توانید همهٔ گزینه‌های تکراری آرایه را کنار بگذارید و در هر خط فقط `--stage ... --output ./results_v8/paper_ml100k` بدهید. برای Bash از همین شکل استفاده کنید؛ syntax آرایهٔ PowerShell قابل انتقال به Bash نیست.

معادل خودکار مراحل:

```powershell
python paper_v8.py --stage all @V8Paper --include-temporal
```

all ابتدا scan و TEST اصلی را می‌گیرد، سپس تحلیل حساسیت و آزمایش‌های تکمیلی را اجرا می‌کند. بنابراین اگر sensitivity را برای تصمیم‌گیری دربارهٔ search می‌خواهید، ترتیب دستیِ scan→lossstudy→مطالعهٔ تازه→test را انتخاب کنید. اجرای دوبارهٔ all بعد از مسیر دستی، jobهای معتبرِ کامل را reuse می‌کند؛ اجرای هر دو مسیر از ابتدا لازم نیست.

| مرحلهٔ تکمیلی | پرسشی که پاسخ می‌دهد | محدودیت تفسیر |
|---|---|---|
| ablation | حذف یک مؤلفه چه اثری دارد؟ | config مرجع منجمد؛ TEST برای تنظیم دوباره نیست |
| ablation-fixed | اثر مؤلفه در K ثابت چیست؟ | K ثابت پیش‌فرض۲۰ است؛ با K منتخب یکسان فرض نشود |
| kstudy | حساسیت به K چیست؟ | برای منحنی چندنقطه‌ای، از ابتدا چند K تعریف کنید |
| sparsity | با حفظ بخشی از TRAIN چه می‌شود؟ | VAL/TEST ثابت؛ cold-start کامل نیست |
| convergence | رفتار TRAIN/VAL با schedule مرجع چیست؟ | آموزش تشخیصی تازه؛ منحنی TEST در epoch نمی‌سازد |
| temporal | تحت ترتیب زمانی چه می‌شود؟ | scan/TEST و جمعیت جدا؛ با split تصادفی جمع نشود |
| calibration | انتخاب با VAL NLL چه تغییری می‌دهد؟ | مطالعهٔ جدا روی common؛ paper، متد۷ point-only دارد |
| density | رفتار در تراکم‌های synthetic چیست؟ | جمعیت مشاهده‌شده و heldout می‌تواند عوض شود |

برای شروع کوچک‌ترِ بخش مقاله، از ابتدا خروجی تازه و مثلاً پنج split، دو seed و سه K تعریف کنید:

```text
python paper_v8.py --stage plan --profile paper --datasets ml-100k --data-root ./datasets --output ./results_v8/paper_reduced --device cuda --split-seeds 4101,4102,4103,4104,4105 --model-seeds 11,22 --k 10,20,50 --search-jobs-per-method 6
python paper_v8.py --stage all --output ./results_v8/paper_reduced --include-temporal
```

این خروجی مطالعهٔ۲۰splitی پیش‌فرض نیست. نقش model seed، تکرار آموزش در همان split است؛ آن‌ها را تکرار مستقل داده حساب نکنید.

## ۷. احتمال، synthetic و همهٔ دیتاست‌ها

روی مطالعهٔ common که scan آن انجام شده است:

```text
python paper_v8.py --stage calibration --output ./results_v8/tuned_light
```

انتخاب با VAL NLL، جدا از انتخاب اصلی با RMSE گزارش می‌شود. در paper/aemc-paper، مرحلهٔ calibration به دلیل point-only بودن متد۷ کنار گذاشته می‌شود؛ اجرای آن نقش‌های احتمالاتی مستقل نمی‌سازد. در all مربوط به synthetic، مرحلهٔ density به‌صورت خودکار اضافه می‌شود. temporal فقط برای خانوادهٔ MovieLens اجرا می‌شود.

معادل اجرای همهٔ شش دیتاست:

```powershell
python paper_v8.py --stage plan --profile paper --datasets all --data-root $DataRoot --output ./results_v8/paper_all --device cuda --host-budget-mb 512 --ssd-budget-gb 8
python paper_v8.py --stage all --output ./results_v8/paper_all --include-temporal
```

زمان ML-100K را در۶ ضرب نکنید. پیش از paper_all، برای هر دیتاست smoke و پایلوت مستقل، در خروجی جدا بگیرید. baselineهای مجاز طبق ماتریس روش/دیتاست قبلی تعیین می‌شوند؛ غیرفعال‌بودن یک baseline ناسازگار در synthetic حذف یکی از هشت روش پیشنهادی نیست.

## ۸. RAM، GPU، SSD، resume و seed

- اجرای پیش‌فرض، jobهای جدا و پشت سر هم است. وزن و cache مدل تمام‌شده پیش از مدل بعدی آزاد می‌شود؛ allocated و reserved حافظهٔ GPU جدا ثبت می‌شوند. reserved در ابزار سیستم لزوماً tensor زنده نیست.
- روی لپ‌تاپ RTX3050 با۶GB VRAM و۸GB RAM، ابتدا light را با `--host-budget-mb 512 --ssd-budget-gb 8` و فضای آزاد SSD کافی امتحان کنید. این مقادیر سهم بودجهٔ ذخیره‌اند، نه سقف کل RAM برنامه یا نتایج نهایی.
- offload محل ذخیره را عوض می‌کند؛ dtype، loss و تعداد update منطقی را خودکار سبک نمی‌کند. CPU fallback و دستگاه واقعی باید در مقایسهٔ هزینه ثبت شوند.
- backward تاریخچهٔ gated ماتریس چگال U×I نمی‌سازد. snapshot اولیهٔ مشترک replay نیز در حد بودجهٔ RAM، و در غیر آن روی SSD نگه داشته می‌شود.
- همان سورس، داده، seed، بودجه و output را برای ادامه نگه دارید. برای مراحل آموزش `--resume` خودکار است؛ گزینهٔ صریح هم پذیرفته می‌شود. job کامل آموزش مجدد نمی‌بیند؛ job جزئی تا مرز معتبر قابل بازیابی است.
- Method8 trainer اصلی را حفظ می‌کند: کاندیدای کامل reuse می‌شود؛ optimizer داخل epoch مانند هفت روش دیگر بازیابی نمی‌شود.
- seed تقسیم داده با seed آموزش فرق دارد. `--manual-seed 17`، model seed و search seed را تعیین می‌کند و با `--model-seeds` همزمان استفاده نمی‌شود. `--deterministic` از ابتدا در recipe انتخاب می‌شود و برابری بیتی بین سخت‌افزارها را تضمین نمی‌کند.
- برای عوض‌کردن تنظیمات علمی، `--output` تازه بسازید. برنامه حالت smoke را روی recipe پژوهشی قبلی تحمیل نمی‌کند. CPU/GPU، threads و بودجهٔ ذخیره از گزینه‌های قابل تغییر محل اجرا هستند.

مثال seed دستی، در مطالعهٔ تازه:

```text
python paper_v8.py --stage plan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/seed17 --manual-seed 17 --split-seeds 2026 --deterministic --device cuda
python paper_v8.py --stage develop --output ./results_v8/seed17
```

## ۹. تخمین زمان؛ معادل بخش زمان‌بندی نسخه۷

جدول زمان نسخه۷ در فایل اصلی‌اش حفظ شده است، اما بنچمارک اندازه‌گیری‌شدهٔ نسخه۸ روی RTX3050/4090 یا Colab نیست. نسخه۸ همان ابزار شمارش برنامه و تخمین مرجع را دارد. مسیر فایل مرجعِ ابزار اصلاح شده تا نصب تمیز نیز بتواند آن را اجرا کند:

```text
python estimate_v8.py --profile common --search-jobs-per-method 6 --max-epochs 120 --steps-per-epoch 2 --baseline-max-epochs 200 --output ./results_v8/estimates/common_light.json
python estimate_v8.py --profile paper --output ./results_v8/estimates/paper.json
python estimate_v8.py --profile common --search-jobs-per-method 6 --max-epochs 120 --steps-per-epoch 2 --baseline-max-epochs 200 --calibration ./results_v8/fixed_light/ml-100k/main/candidate_validation_runs.csv --output ./results_v8/estimates/local_light.json
```

فرمان سوم از زمان‌های پایلوت دستگاه خودتان استفاده می‌کند. مرجع پیش‌فرض از پایلوت قدیمی RTX5090 است؛ برای Method8 در صورت نبود اندازه‌گیری، placeholder وجود دارد. بازهٔ۰٫۵تا۴برابر، فرض برنامه‌ریزی است و فاصلهٔ اطمینان آماری نیست. تفاوت بودجه، early stopping، ظرفیت، SSD و fallback می‌تواند از آن فراتر رود. ابزار برای دیتاست‌های دیگر CSV زمان همان دیتاست را لازم دارد.

| بخش | علت اصلی زمان |
|---|---|
| plan | ساخت گرید و شمارش؛ آموزش ندارد |
| smoke / pilot | تعداد روش‌ها و initializationهای لازم، epoch و مسیر CPU/GPU |
| scan | تعداد واقعی کاندیدا × K × split × model seed، با استثنای ظرفیت مستقل autoencoderها |
| test / final | یک config/K/init منتخب برای هر روش و split، با تکرار seedهای آموزش |
| lossstudy / ablation | تعداد ضرایب/حذف‌ها؛ وابسته به سازوکار فعال winner |
| kstudy / sparsity / convergence | بازآموزی‌های تشخیصی جدا |
| temporal | scan+TEST مستقل دیگر |
| report / curves / audit / status | بدون آموزش؛ هزینهٔ CPU، خواندن آرشیو و ذخیرهٔ شکل |

## ۱۰. TEST و نمودارها دقیقاً کجا هستند؟

بعد از test موفقِ مطالعهٔ tuned، زیر `results_v8/tuned_light/ml-100k/`:

| مسیر | نقش |
|---|---|
| `search/hyperparameter_validation_all.csv` | تمام کاندیداهای VAL؛ بدون ستون TEST |
| `search/validation_winners.csv` | برندهٔ هر روش/split |
| `search/best_configurations.json` | config منتخب و منشأ postfit |
| `search/publication/tables/search_parameter_coverage.csv` | تعداد مقادیر واقعاً آزموده‌شدهٔ هر پارامتر |
| `final_locked_test_all.csv` | هر آموزش نهایی و معیارهای TEST |
| `final_locked_test_summary.csv` | خلاصهٔ اصلی مقایسهٔ TEST |
| `final_selection_provenance.json` | ثبت انتخاب با VAL |
| `publication/research_tables.xlsx` | جدول‌های اصلی و تشخیصی |
| `publication/figures/selected_test_rmse.png` | نمودار اصلی TEST |
| `publication/figures/` | PNG با۴۰۰dpi و PDF/SVG برداری |
| `publication/tables/objective_coefficients.csv` | ضرایب ثبت‌شده؛ فعالیتشان به سازوکار وابسته است |
| `publication/tables/ensemble_diversity.csv` | هم‌بستگی residual اعضای متد۷، در صورت وجود آرشیو اعضا |
| `MANUSCRIPT_CHECKLIST.md` | وضعیت تکمیل، شکل‌ها و مراحل باقی‌مانده |

شکل‌های search متعلق به VAL هستند؛ شکل‌های نهایی از TEST موجود ساخته می‌شوند. منحنی epoch باید TRAIN/VAL باشد؛ TEST برای انتخاب epoch اندازه‌گیری نمی‌شود. پوشش interval و risk–coverage با RMSE تمام TEST یک معیار نیستند. صفرشدن OOB نیز به‌تنهایی مزیت کافی نسبت به baseline بریده‌شده نیست.

معادل دو فرمان report و curves نسخه۷، با استفاده از recipe ذخیره‌شده:

```text
python paper_v8.py --stage report --output ./results_v8/tuned_light --plots
python paper_v8.py --stage curves --output ./results_v8/tuned_light --plots
python paper_v8.py --stage status --output ./results_v8/tuned_light
```

معادل مستقیم نیز حفظ شده است:

```text
python run_v8.py report --datasets ml-100k --output ./results_v8/tuned_light
python run_v8.py curves --datasets ml-100k --output ./results_v8/tuned_light
```

این‌ها آموزش تازه نمی‌دهند. curves فقط history موجود را رسم می‌کند؛ اگر RMSE کامل TRAIN ثبت نشده باشد، برای آن از convergence استفاده کنید. برای شکل سفارشی از CSV، ابزار `replot_v8.py` و بخش۶ راهنمای جامع باقی‌اند.

## ۱۱. انتخاب مسیر و کفایت شواهد

۱. نصب و verify؛ ۲. smoke؛ ۳. پایلوت همان دستگاه برای زمان/حافظه؛ ۴. plan و scan مطالعهٔ light؛ ۵. حساسیت VAL در صورت استفاده برای توسعه؛ ۶. TEST تنظیمات منجمد؛ ۷. مطالعهٔ بزرگ‌تر با خروجی تازه و آزمایش‌های تکمیلی؛ ۸. report، audit و status.

جدول اصلی TEST، baselineهای تنظیم‌شده، تکرار داده/آموزش با تفسیر درست استقلال، دیتاست‌های متناسب با ادعا، trained ablation، کنترل decoder/postprocessing، حساسیت VAL و هزینه/حافظه لازم‌اند. برای ادعای probability، baseline احتمالاتی هم لازم است. audit حضور و سازگاری فایل‌ها را بررسی می‌کند؛ تأیید نوآوری، SOTA یا پذیرش Q1 نیست.

گیت اعتماد نسخه۸ طبق config اصلی فعال است؛ اجرای معمول به روشن‌کردن یک سوییچ پنهان نیاز ندارد. مقایسهٔ خاموش/روشن و شدت هندسه در مسیر ablation و search گزارش می‌شود. تغییر gate یا لاس بعد از مشاهدهٔ TEST، بخشی از توسعهٔ تازه است و نتیجهٔ تأییدی جدا می‌خواهد.
