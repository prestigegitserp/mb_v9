# راهنمای اجرایی V7 development

این شاخه بازسازی تازه از `v6@7fbce77f0d07a057083e2bab68327780986cafda` است. انتشار V7 یا merge انجام نشده است. فایل‌های V1 تا V6 تغییر نمی‌کنند. فرمان‌های V7 صریح‌اند؛ `run_v9.py` و فایل `VERSION` برای حفظ رفتار نسخه‌های قبل دست نخورده‌اند.

## ۱. دریافت در پوشهٔ خالی و بررسی CUDA

در PowerShell و داخل پوشهٔ واقعاً خالی:

```powershell
git clone --branch v7-development --single-branch https://github.com/prestigegitserp/mb_v9.git .
python -m pip install -r requirements_v7.txt
python verify_v7.py
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

اگر CUDA برابر False باشد، نصب PyTorch این محیط را با دستور مناسب سیستم خود از [راهنمای رسمی PyTorch](https://pytorch.org/get-started/locally/) اصلاح کنید. نصب بسته‌ها در یک محیط Python دیگر، این محیط را اصلاح نمی‌کند. `--v7-device cuda` در نبود CUDA خطا می‌دهد؛ این حالت بی‌صدا CPU نمی‌شود. افتادن یک کاندیدای CUDA به CPU پس از OOM/خطای عددی، با علت در فایل منابع همان کاندیدا ثبت می‌شود.

آپدیت بعدی همین پوشه:

```powershell
git switch v7-development
git pull --ff-only origin v7-development
python -m pip install -r requirements_v7.txt
python verify_v7.py
```

اگر قبلاً V7 را روی Windows گرفته‌اید و پس از pull فقط کنترل بایتی Method10 خطا داد، ممکن است فایل قدیمی با CRLF مانده باشد. **تنها اگر این فایل اصلی را عمداً محلی ویرایش نکرده‌اید**، نسخهٔ دقیقِ ثبت‌شده در Git را دوباره checkout کنید و بررسی را تکرار کنید:

```powershell
git restore --source=HEAD --worktree -- vendor_method10/method10.py
python verify_v7.py
```

clone تازه این مشکل را ندارد؛ قاعدهٔ انتقال مخصوص همین فایل، بایت‌های ZIP را روی Windows هم حفظ می‌کند.

برای داده‌ها می‌توانید پوشهٔ نسخهٔ قبلی را مستقیماً استفاده کنید:

```powershell
$DataRoot = 'C:\Users\ASUS\Desktop\RUN\mb_v9_git_v5\datasets'
```

در مثال‌های بعدی، اگر داده در همین پروژه است، `$DataRoot` را با `./datasets` عوض کنید. پسوند دستور قبلی `npython` اشتباه تایپی بود؛ نام فرمان `python` است.

## ۲. سه نوع اجرا که نباید مخلوط شوند

| هدف | فرمان/پروفایل | انتخاب هایپرپارامتر | کاربرد |
|---|---|---|---|
| پایلوت ثابت مشابه V6 | `main` با `common` | تنظیم پیش‌فرض؛ فقط انتخاب epoch با validation | توسعه و مقایسهٔ ثابت نسخه‌ها |
| پایلوت تنظیم‌شده | `develop` با `common` | جست‌وجوی محدود، سپس refit و TEST منتخب‌ها | مقایسهٔ توسعه‌ای منصفانه با بودجهٔ معلوم |
| مطالعهٔ مقاله | `paper`، ابتدا `scan` سپس `final` و تحلیل‌ها | انتخاب جداگانه برای هر روش و outer split | چند split، چند seed، آرشیو و گزارش مقاله |

`common` مانند پایلوت ارسالی شما تقسیم per-user تقریباً ۸۰/۱۰/۱۰ با اصلاح پشتیبانی دارد؛ روی ML-100K/2026 تعداد TRAIN/VAL/TEST برابر ۸۰۰۵۳/۹۹۶۷/۹۹۸۰ است. ۶۰٪ validation برای انتخاب، ۲۰٪ برای probability fit و ۲۰٪ برای holdout احتمالاتی رزرو می‌شود. این سه نقش برای همهٔ مدل‌ها تعریف می‌شوند؛ Method7 فقط در این پروتکل می‌تواند از بخش‌های احتمالاتی استفاده کند.

`development` و `paper` روی دیتاست‌های واقعی از `aemc-paper` استفاده می‌کنند: تقسیم global برابر ۸۵٫۵/۴٫۵/۱۰، تمام validation برای انتخاب، و Method7 در حالت point-only. تغییر `common` به این پروتکل، TEST را عوض می‌کند؛ کاهش عدد میان آن‌ها به‌تنهایی شاهد بهترشدن مدل نیست. پروفایل synthetic مطابق فایل تنظیمات خودش است.

## ۳. فرمان سادهٔ مقایسه‌ای شما

```powershell
python run_v7.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v7/pilot_common --v7-device cuda --v7-protocol common --no-epoch-train-rmse --resume
```

این فرمان اکنون ۱۵ خانواده و ۲۱ آموزش کاندیدا دارد: هشت روش شما، هفت رقیب، و سه initialization برای هر MBMF. تنظیمات جدید V7 با V6 فرق دارند؛ پروتکل مشترک و seed یکسان، مقایسهٔ معتبرِ تغییر نسخه را ممکن می‌کند. هر مدل بهترین epoch خود را با validation انتخاب می‌کند، ولی این فرمان جست‌وجوی هایپرپارامتر نیست.

`--no-epoch-train-rmse` فقط محاسبهٔ اضافهٔ خطای کامل TRAIN در هر tick را حذف می‌کند. loss و مراحل optimizer عوض نمی‌شوند؛ خروجی نهایی TRAIN/VAL/TEST و پیش‌بینی‌های منتخب همچنان ساخته می‌شوند. اگر منحنی کامل TRAIN در هر tick لازم است، این گزینه را بردارید.

نسخهٔ سبک‌تر با سقف آموزش کمتر:

```powershell
python run_v7.py main --datasets ml-100k --data-root $DataRoot --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v7/pilot_short --v7-config ./examples/v7/common/ml-100k.json --v7-device cuda --v7-max-epochs 120 --v7-steps-per-epoch 2 --v7-baseline-max-epochs 200 --no-epoch-train-rmse --resume
```

این بودجه با پایلوت قدیمی یکسان نیست. برای مقایسهٔ علمی، سقف epoch، تعداد update، اندازهٔ مدل و زمان واقعی را همراه RMSE گزارش کنید. `--quick` فقط آزمون سه‌epochی سالم‌بودن اجراست و معیار دقت مقاله نیست.

## ۴. پایلوت تنظیم‌شده؛ پاسخ به ابهام هایپرپارامتر

```powershell
python paper_v7.py --stage plan --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v7/tuned_common
python paper_v7.py --stage develop --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v7/tuned_common --device cuda
```

`develop` معادل اجرای `scan` بدون `--scan-only` است. برنامه کاندیداها را روی TRAIN آموزش می‌دهد، پارامتر و معماری و K را با validation انتخاب می‌کند، تنظیم منتخب هر روش/outer split را ذخیره می‌کند، سپس **خودکار** `final` و گزارش منتخب‌ها را تولید می‌کند. Excel و نمودارهای نهایی هم تولید می‌شوند؛ خروجی فقط «تحقیق دربارهٔ هایپرپارامتر» نیست.

پروفایل common یک split، یک seed و K=20 دارد. سقف پیش‌فرض هشت job برای هر روش است؛ برنامهٔ فعلی ۱۲۴ آموزش اسکن دارد، نه هشت آموزش کل. Method2 داخل هر کاندیدا OOF experts هم آموزش می‌دهد. AutoRec و AEMC-NE به K وابسته نیستند و برای Kهای تکراری دوباره آموزش نمی‌بینند. فایل plan تعداد بیرونی آموزش‌ها را نشان می‌دهد؛ آموزش‌های داخلی Method2 در trace و اطلاعات بودجه ثبت می‌شوند. تنظیم‌های موفقِ validation در V6 نیز با عنوان `prior_validated_v6` وارد همین رقابت شده‌اند؛ budgetهای CLI بر آن‌ها هم اعمال می‌شوند. اعداد تاریخیِ آن فایل متعلق به پروتکل دیگری‌اند و نتیجهٔ تازهٔ common محسوب نمی‌شوند.

برای کاهش بودجهٔ توسعه:

```powershell
python paper_v7.py --stage develop --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v7/tuned_small --device cuda --search-jobs-per-method 4 --max-epochs 120 --steps-per-epoch 2 --baseline-max-epochs 200
```

سقف چهار job الزاماً همهٔ coefficientها را تغییر نمی‌دهد؛ گرید محدود هرگز بهینه‌سازی تضمینی نیست. `search_parameter_coverage.csv` دقیقاً نشان می‌دهد چه پارامتری چند مقدار متفاوت داشته و کدام در بودجهٔ بررسی‌شده ثابت مانده است. برای بررسی ضرایب loss، سقف پیش‌فرض هشت و مرحلهٔ `lossstudy` مناسب‌ترند. صفرکردن سقف (`--search-jobs-per-method 0`) جست‌وجوی بدون سقف است و هزینهٔ زیادی دارد.

**پس از اسکن، `main` نزنید تا برنده‌ها را مصرف کند؛ `main` برنده‌ها را بارگذاری نمی‌کند.** فرمان درست `final` با همان config، seedها، Kهای اسکن و پروتکل است، یا همان `develop` که هر دو مرحله را انجام می‌دهد.

## ۵. ترتیب کامل مقاله روی ML-100K

بودجه و نام پوشه را از ابتدا ثابت کنید. مثال جامع با ۲۰ outer split، سه model seed و Kهای ۱۰/۲۰/۵۰:

```powershell
python paper_v7.py --stage plan --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k
python estimate_v7.py --profile paper --dataset ml-100k

python paper_v7.py --stage scan --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage final --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage lossstudy --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage ablation --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage ablation-fixed --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage kstudy --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage sparsity --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage convergence --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage temporal --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda
python paper_v7.py --stage report --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k
python paper_v7.py --stage audit --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k
```

تمام این فرمان‌ها را می‌توان با یک دستور اجرا کرد:

```powershell
python paper_v7.py --stage all --profile paper --datasets ml-100k --data-root $DataRoot --output ./results_v7/paper_ml100k --device cuda --include-temporal
```

این کار بسیار سنگین است. برای شروع مقاله می‌توانید به‌طور صریح `--split-seeds 4101,4102,4103,4104,4105 --model-seeds 11,22 --k 20` را به **تمام** مرحله‌های همان مطالعه اضافه کنید. این مطالعهٔ کوچک‌تر را به‌عنوان ۲۰ split یا بازتولید کامل جدول مقاله معرفی نکنید. بعداً تعداد split/seed/K را در همان پوشه تغییر ندهید؛ برای مطالعهٔ بزرگ‌تر خروجی تازه انتخاب کنید.

اسکن باید پیش از گزارش اصلی تنظیم‌شده باشد. `final` از همان TRAIN و همان validation استفاده می‌کند، TRAIN+VAL را ادغام نمی‌کند، و قاعدهٔ از پیش تعیین‌شدهٔ انتخاب epoch را حفظ می‌کند. TEST در انتخاب پارامتر استفاده نمی‌شود. میانگین seedهای آموزش ابتدا داخل هر outer split محاسبه می‌شود؛ سپس میانگین و SD بین splitها. یک split، خطای معیار یا آزمون معناداری بین‌split معتبر ایجاد نمی‌کند.

## ۶. همهٔ دیتاست‌ها و نقش تحلیل‌ها

```powershell
python paper_v7.py --stage plan --profile paper --datasets all --data-root $DataRoot --output ./results_v7/paper_all
python paper_v7.py --stage all --profile paper --datasets all --data-root $DataRoot --output ./results_v7/paper_all --device cuda --include-temporal
```

قالب‌های مستقل `common`، `development` و `paper` برای `synthetic`، `ml-100k`، `ml-1m`، `ml-10m`، `jester` و `bookcrossing` در `examples/v7` هستند. داده‌های واقعی باید آماده باشند. برای synthetic تولید داده با seed انجام می‌شود. مرحلهٔ `density` فقط برای synthetic و `temporal` فقط برای MovieLens اجرا می‌شود. جدول سازگاری رقیب/دیتاست حفظ شده؛ مثلاً حذف یک رقیب از synthetic به معنی خرابی CUDA نیست.

| مرحله | آنچه ثابت یا تغییر می‌کند | مرجع گزارش |
|---|---|---|
| scan | آموزش TRAIN؛ مقایسهٔ config/معماری/K و ضرایب loss روی validation | برنده‌ها و کل تاریخچهٔ جست‌وجو |
| final | config منتخب هر روش/split، seedها و پروتکل قفل می‌شوند | جدول اصلی TEST |
| ablation | شروع از همان برنده؛ حذف یک مؤلفه بدون انتخاب مجدد با TEST | اثر مؤلفه در تنظیم منتخب |
| ablation-fixed | تحلیل حذف مؤلفه با K ثابتِ اعلام‌شده؛ ثانویه | مقایسهٔ ظرفیت هم‌تراز |
| lossstudy | تغییر یک ضریب در هر بار از تنظیم منتخب؛ فقط validation | نمودار حساسیت loss، بدون TEST |
| kstudy | تغییر K از نقطهٔ منتخب؛ نتیجهٔ تشخیصی | منحنی حساسیت K؛ K اصلی از scan می‌آید |
| sparsity | کاهش دادهٔ TRAIN و کنترل یکسان بین روش‌ها | پایداری در کم‌داده بودن |
| convergence | همان schedule منتخب با ثبت بیشتر TRAIN/VAL | همگرایی؛ بدون انتخاب TEST |
| calibration | مطالعهٔ جداگانهٔ probability در common و با نقش‌های disjoint | NLL/RPS/coverage؛ با جدول اصلی مخلوط نشود |
| temporal | اسکن و انتخاب جداگانه در تقسیم زمانی | مطالعهٔ زمانی مستقل |
| report | بازسازی Excel/شکل از داده‌های ذخیره‌شده؛ بدون آموزش | امکان اصلاح ظاهر مقاله |

برای calibration واقعی Method7 از پروفایل common استفاده کنید؛ پروفایل `aemc-paper` به‌طور صریح point-only است و مرحلهٔ احتمالاتی را رد می‌کند. Method8 معماری و loss اصلی خود را دارد؛ برای آن ablation ساختگیِ مؤلفه‌های Methods1–7 تولید نمی‌شود. داده و زمان و پیش‌بینی Method8 در جدول اصلی و تحلیل‌های سازگار باقی می‌مانند.

## ۷. Resume و نگهداری دیسک

همان فرمان و همان پوشه را با `--resume` تکرار کنید؛ recipeهای `paper_v7.py` این گزینه را برای آموزش خودکار اضافه می‌کنند.

- job کامل: آموزش تکرار نمی‌شود؛ گزارش قابل بازسازی است.
- job نیمه‌تمام: کاندیداهای تمام‌شده از state و train_info معتبر بازیابی می‌شوند.
- میان آموزش Methods1–7 در مسیر جدید و تاریخی V3/V4/V5.2 و رقبا در policy پشتیبانی‌شده: مدل، optimizer، بهترین state، patience، شمارنده‌ها و RNG بازیابی می‌شوند. پیش‌فرض هر پنج epoch ذخیره می‌شود؛ حداکثر چهار epoch کامل ممکن است تکرار شود. `--v7-recovery-interval 1` ذخیره در هر epoch است و I/O بیشتری دارد.
- Method2: mean-phase هر OOF expert جدا checkpoint دارد؛ مرحلهٔ کوچک router می‌تواند دوباره fit شود. probability/response fit کوچک نیز ممکن است دوباره انجام شود.
- Method8: trainer اصلی تغییر نکرده است؛ کاندیدای کامل reuse می‌شود، ولی کاندیدای قطع‌شده از ابتدای trainer اصلی شروع می‌شود. optimizer-resume داخل آن ادعا نمی‌شود.

هویت resume به کد، دادهٔ ورودی، config علمی، بودجه و seedها وابسته است؛ تغییر GPU/CPU، مسیر خروجی یا ظاهر شکل به‌تنهایی آزمایش علمی تازه نیست. تغییر داده، loss یا budget در پوشهٔ قبلی خطای صریح می‌دهد. قفل سیستم‌عامل مانع دو نویسندهٔ همزمان می‌شود و پس از crash آزاد می‌شود؛ هیچ سیگنالی به PID دیگر نمی‌فرستد. وجود فایل lock خالی پس از پایان اجرا طبیعی است. زمان آموزشِ کاندیدای reuseشده با زمان چندمیلی‌ثانیه‌ای خواندن cache جایگزین نمی‌شود؛ ستون‌های reuse و زمان تلاش جاری جدا ثبت می‌شوند.

در پایان job کامل، recoveryهای موقت حذف می‌شوند. در اسکن، امتیاز و config و history همه باقی می‌ماند؛ پیش‌بینی‌های حجیم فقط برای رهبران فعلی validation نگه داشته می‌شوند. final، آرشیو کامل منتخب‌ها و Excel/شکل‌های نهایی را می‌سازد. lossstudy تنها workbook تجمیعی می‌سازد و برای هر trial یک Excel جدا انباشته نمی‌کند. `--ssd-budget-gb` سقف cache موقت است؛ سقف کل نتایج پژوهش و گزارش مقاله نیست.

پاکسازی پس از پایان موفق worker در parent هم تکرار می‌شود؛ job قطع‌شده یا گزارش ناموفق recovery خود را از دست نمی‌دهد. زمان checkpointهای قبلیِ expertهای OOF و warm-start در جمع زمان کاندیدای ادامه‌یافته منظور می‌شود؛ reuse یک کاندیدای کامل همچنان از refit یک فاز کوچک جدا ثبت می‌شود.

## ۸. حافظه، زمان و بازتولید

نمونهٔ تنظیم منابع برای RAM سیستم ۱۶ گیگابایت:

```powershell
python paper_v7.py --stage develop --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v7/tuned_common --device cuda --host-budget-mb 512 --ssd-budget-gb 8 --cache-dir ./v7_ssd_cache
```

فرایند جدا برای هر job پیش‌فرض است، نه اجرای موازی چند job روی یک GPU. بین کاندیداها، مرجع مدل قبلی، gradient و cache آزاد می‌شود؛ دادهٔ splitهای قبلی روی GPU انباشته نمی‌شود. offload فقط مکان نگهداری tensorهای ذخیره‌شده را عوض می‌کند؛ dtype، objective، batch منطقی و تعداد update خودکار تغییر نمی‌کنند. replay روی CPU ممکن است اختلاف گردکردن داشته باشد و برابری بیت‌به‌بیت CPU/CUDA ادعا نمی‌شود.

runner فقط دادهٔ tensorشدهٔ یک split را نگه می‌دارد. raw data یک بار خوانده می‌شود و برای splitهای دیگر، نقش train/validation/test هر تعامل به‌صورت int8 روی SSD نگه داشته می‌شود؛ ترتیب، نوع و مقدار ردیف‌ها حفظ می‌شود. این cache از همان سهمیهٔ موقت SSD کم می‌شود. اگر سهمیه برای نقش‌ها کافی نباشد، split به‌صورت قطعی دوباره ساخته می‌شود. فایل‌های CSV یکسانِ split/ID map/نقش validation بین jobها با hardlink اشتراک دارند؛ اگر فایل‌سیستم اجازه ندهد، copy با ثبت علت جایگزین می‌شود. مسیر هر job همچنان موجود است. پوشهٔ `_shared_data` جزو نتایج مطالعه است و تا زمانی که jobها به آن وابسته‌اند نگهش دارید.

تشخیص RAM به محدودیت container هم توجه می‌کند. در حالت auto، کمبود RAM می‌تواند ذخیرهٔ activation را روی SSD ببرد؛ این قابلیت، نیاز اولیهٔ خودِ raw dataset یا پارامترهای مدل به RAM را حذف نمی‌کند. اگر cache موقت پر شود خطای روشن می‌گیرید؛ ظرفیت دیسک پرشده با تغییر loss یا batch پنهان نمی‌شود. مسیر داده و خروجی پیش از ایجاد worker مطلق می‌شود، بنابراین اجرای script از پوشه‌ای دیگر هم همان مسیرهای داده‌شده را به کار می‌برد.

`allocated`، `reserved` و مصرف کل device یکسان نیستند؛ مصرف device-wide می‌تواند متعلق به فرایندهای دیگر هم باشد. `V7_RESOURCE_LIFECYCLE.jsonl`، `V7_JOB_PROCESS.json` و `DEVICE_AND_STORAGE.json` علت و میزان حافظه را ثبت می‌کنند. آزادکردن cache به‌تنهایی tensor زنده را آزاد نمی‌کند.

```powershell
python estimate_v7.py --dataset ml-100k --profile common
python estimate_v7.py --dataset ml-100k --profile paper
python estimate_v7.py --dataset ml-100k --profile common --calibration ./results_v7/pilot_common/ml-100k/main/candidate_validation_runs.csv
```

دو فرمان اول از زمان‌های پایلوت ارسالی روی **RTX 5090** استفاده می‌کنند؛ **بنچمارک RTX4090 نیستند**. مدل دستگاه در ۲۰ فایل `DEVICE_AND_STORAGE.json` پایلوت ثبت شده است. بازهٔ برنامه‌ریزی ۰٫۵ تا ۴ برابر مرجع، فرض است و CI آماری نیست. CSV کالیبراسیون به‌تنهایی نام سخت‌افزار را اثبات نمی‌کند؛ باید از اجرای سیستم خودتان بیاید. زمان Method8 تا وقتی روی سیستم شما اندازه‌گیری نشود یک placeholder چهل‌ثانیه‌ای است. برای common فعلی:

| مرحله | تعداد آموزش بیرونی | بازهٔ فرضی زمان |
|---|---:|---:|
| main | 21 | 7–57 دقیقه |
| scan | 124 | 0.64–5.15 ساعت |
| final | 15 | 4–34 دقیقه |
| ablation با مدل‌های پیش‌فرض | 103 | 0.63–5.01 ساعت |
| lossstudy با مدل‌های پیش‌فرض | 222 | 1.35–10.84 ساعت |
| همهٔ مرحله‌های درخواستی، بدون temporal | 656 | 3.75–30.02 ساعت |

تعداد ابلیشن و lossstudy به معماریِ برنده وابسته است؛ این اعداد برنامهٔ پیش‌فرض‌اند. سقف epoch، OOF، SSD و CPU fallback می‌تواند این بازه‌ها را بشکند. RTX4090 حافظهٔ VRAM برابر ۲۴ گیگابایت دارد؛ RAM سیستم ۱۶ گیگابایت منبع دیگری است.

پروفایل کامل paper حدود ۳۴۰۲۰ آموزش اسکن و ۹۰۰ آموزش نهایی برای ML-100K دارد؛ زمان اسکن مرجع حدود ۳۹۰ ساعت و بازهٔ فرضی حدود ۱۹۵–۱۵۶۰ ساعت است. برنامهٔ پیش‌فرض همهٔ مرحله‌ها بدون temporal حدود ۶۶۹۶۰ آموزش بیرونی و بازهٔ فرضی ۳۸۹–۳۱۰۹ ساعت دارد؛ این زمان برای همهٔ شش دیتاست نیست. قبل از شروع، plan و اندازه‌گیری پایلوت روی سخت‌افزار خود را مبنا قرار دهید.

برای seed دستی:

```powershell
python paper_v7.py --stage develop --profile common --datasets ml-100k --data-root $DataRoot --output ./results_v7/seed17 --device cuda --manual-seed 17 --deterministic
```

`--manual-seed` seed مدل و جست‌وجوی تصادفی محدود را مشخص می‌کند. `--split-seeds` جداست. همزمان‌دادن `--manual-seed` و `--model-seeds` خطاست. برای چند تکرار از `--model-seeds 11,22,33` استفاده کنید. deterministic در یک محیط ثابت به بازتولید کمک می‌کند؛ تضمین برابری همهٔ GPUها، نسخه‌های PyTorch و CPU/CUDA نیست و عملیات CUDA فاقد پیاده‌سازی deterministic باید خطا بدهند.

## ۹. خروجی‌های مقاله و رسم دوباره از Excel

در ریشهٔ هر دیتاست:

- `search/best_configurations.json`: config منتخب هر مدل و outer split، بدون TEST.
- `search/hyperparameter_validation_all.csv`: همهٔ کاندیداهای validation و زمان/بودجه‌شان.
- `search/publication/tables/hyperparameters_long.csv` و `search_parameter_coverage.csv`: مقادیر بررسی‌شده و پوشش واقعی بودجه.
- `final_locked_test_summary.csv`: جدول اصلی TEST؛ seed ابتدا داخل split میانگین می‌شود.
- `publication/research_tables.xlsx`: INDEX و جدول‌های قابل بازتحلیل، از جمله ضرایب objective.
- `publication/tables/objective_coefficients.csv` و `objective_components_long.csv`: ضرایب اجرایی و مؤلفه‌های loss ثبت‌شده.
- `lossstudy/loss_sensitivity.xlsx`: دادهٔ حساسیت validation، با anchor و مقدار ضریب.
- `figures/`: PNG با ۴۰۰ DPI، PDF برداری و SVG و دادهٔ پشتیبان شکل.
- جدول‌های مستقل ابلیشن: نام variant حفظ می‌شود و میانگین variantهای مختلف با هم ادغام نمی‌شود.

بازسازی بدون آموزش:

```powershell
python run_v7.py report --datasets ml-100k --output ./results_v7/tuned_common
python replot_v7.py --input ./results_v7/tuned_common/ml-100k/lossstudy/loss_sensitivity.xlsx --sheet all_validation_runs --x value --y val_RMSE --group method --where parameter=reconstruction_weight --kind line --aggregate mean --output ./results_v7/custom_figures/reconstruction_sensitivity
```

`replot_v7.py` علاوه بر شکل، CSV نقاط رسم‌شده و JSON شامل هش منبع و آرگومان‌ها ذخیره می‌کند. به‌همراه shapeها، دادهٔ بازتولید شکل را نگه دارید. SD بین splitها را CI نام‌گذاری نکنید؛ curveهای یک split، تأیید مستقل مقاله نیستند. خود کیفیت ظاهری شکل یا عدد ۰٫۸۷ تضمین پذیرش مقاله نیست؛ پروتکل، نوآوری و آزمون‌های مستقل هم لازم‌اند.

در جدول ضرایب، «ضریب objective»، «بودجهٔ shrinkage optimizer»، «بودجهٔ gradient surgery» و «precision/scale مربوط به prior» جدا مشخص می‌شوند. اجزای prior، auxiliary و reconstruction با نام مستقل در دادهٔ تاریخچه ثبت می‌شوند. RPS تنها در حالت احتمالاتی فعال است؛ lossstudy در آن حالت علاوه بر RMSE، NLL validation را هم ذخیره، تجمیع و رسم می‌کند. ثابت‌ماندن RMSE هنگام تغییر یک ضریب احتمالاتی، به‌تنهایی به معنی بی‌اثر بودن آن بر توزیع احتمالات نیست.

## ۱۰. بازتولید بررسی توسعه‌ای ثبت‌شده

شواهد ۲۷ آموزشِ کوتاه در `docs/v7_evidence` هستند؛ جزئیات و محدودیتشان در [گزارش پژوهشی](V7_RESEARCH_REVIEW_FA.md) آمده است. فرمان زیر بررسیِ فعلی و اصلاح‌شدهٔ سه خانواده را با CUDA تکرار می‌کند. این ابزار جای `develop` و مطالعهٔ مقاله را نمی‌گیرد؛ TEST را ارزیابی نمی‌کند و برندهٔ `final` یک مطالعهٔ دیگر را هم تغییر نمی‌دهد.

```powershell
python experiments/v7_validation_probe.py --data-root $DataRoot --output ./results_v7/validation_diagnostics --epochs 60 --steps 2 --hidden 64 --baselines --method8 --device cuda --resume
```

برای Method3، تغییر lineage به V7 را از اثر خود gate جدا کنید:

```powershell
python experiments/v7_validation_probe.py --data-root $DataRoot --output ./results_v7/method3_diagnostics --epochs 60 --steps 2 --hidden 64 --methods Method3 --families modern,gated --device cuda --resume
```

`reference` پایهٔ فعلی، `prior` تنظیم قبلاً منتخبِ V6، `modern` همان تنظیم روی معماری V7/MSE و `gated` همان مسیر مدرن با gate فعال است. این کنترل‌ها الزاماً همگی بهتر از پایه نیستند؛ نتایج منفی در CSV باقی می‌مانند. اجرای ثبت‌شدهٔ اولیه از driver در commit اعلام‌شده استفاده کرده بود؛ تفاوت کنترل Method3 در گزارش توضیح داده شده است. source/driver/config در پروتکل ذخیره می‌شوند، پس برای تغییر آن‌ها پوشهٔ تازه بسازید.
