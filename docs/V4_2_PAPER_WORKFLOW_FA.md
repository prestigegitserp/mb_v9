# راهنمای اجرای آزمایش‌های مقاله — نسخهٔ ۴٫۲

این راهنما برای `run_v4.py` و شاخهٔ `v4-development` است. `main`، `scan` و `final` سه کار متفاوت‌اند. هیچ commit، push کد یا merge در این تحویل انجام نشده است. برای اجرای نسخه‌های ۱ تا ۳ از entry مستقل همان نسخه استفاده کنید.

## پاسخ روشن به سؤال اسکن

**بله: `scan` در این نسخه فقط یک جدول جست‌وجو تولید نمی‌کند.** برای هر کاندیدا، خروجی TRAIN/VALIDATION، تاریخچه، تنظیمات، دادهٔ قابل بازسازی نمودار و گزارش تولید می‌شود. پس از پایان جست‌وجو نیز گزارش تجمیعیِ برنده‌ها و همهٔ تلاش‌ها نوشته می‌شود. اگر `--scan-only` ندهید، اجرای پیش‌فرض `scan` به ارزیابی نهاییِ برنده‌های ثابت‌شده روی TEST هم ادامه می‌دهد.

`--scan-only` یعنی توقف پیش از TEST، نه حذف فایل‌ها و نمودارهای ولیدیشن. طبیعتاً در این مرحله نباید نمودار TEST تولید شود. `--v4-no-plots` تنها گزینهٔ صریح برای غیرفعال‌کردن رسم است؛ داده‌های گزارش باقی می‌مانند و بعداً با `report` رسم می‌شوند. `--v4-retain-candidates metrics` وزن‌های کاندیداها را حذف می‌کند، اما پیش‌بینی‌ها و دادهٔ گزارش را نگه می‌دارد؛ وزن‌های منتخب مرحلهٔ نهایی همچنان ذخیره می‌شوند.

| دستور | چه چیزی را انتخاب/اجرا می‌کند؟ | نقش TEST |
|---|---|---|
| `main` | تنظیمات JSON/پیش‌فرض؛ انتخاب K و initialization در همان مطالعه با ولیدیشن | فقط پیکربندی منتخب همان مطالعه |
| `scan --scan-only` | جست‌وجوی واقعی تنظیمات، K و initialization با ولیدیشن | ارزیابی نمی‌شود |
| `scan` | جست‌وجو، تثبیت برنده، سپس `final` | پس از تثبیت برنده |
| `final` | بازآموزی با همان تنظیمات/seedها/دادهٔ اسکنِ ثابت‌شده | گزارش نهایی؛ انتخاب تنظیم جدید ندارد |
| `ablation` | حذف مکانیسم از برندهٔ اسکن هر split؛ بقیهٔ تنظیمات و K ثابت | مطالعهٔ مکانیسم؛ برای انتخاب مدل استفاده نشود |
| `ablation-fixed` | همان مطالعه با K از پیش اعلام‌شده و یکسان برای full و تمام حذف‌ها | تحلیل ثانویه |
| `kstudy` | حساسیت به K با ثابت‌کردن سایر تنظیمات منتخب | نمودار توصیفی؛ K را از TEST انتخاب نکنید |
| `sparsity` | حذف فقط یال‌های TRAIN، با ثابت‌ماندن VAL/TEST و تنظیمات منتخب | حساسیت به کمبود داده |
| `convergence` | بازاجرای validation-only با همان زمان‌بندی انتخاب چک‌پوینت | TEST خوانده نمی‌شود |
| `calibration` | تحلیل ثانویهٔ روش ۷ با انتخاب NLL روی selection | جدا از نتیجهٔ اصلیِ انتخاب‌شده با RMSE |
| `ensemble` | کنترل ترکیبیِ دو مدل با وزنِ قبلاً ثابت‌شده روی ولیدیشن | هیچ وزن ترکیبی از TEST انتخاب نمی‌شود |
| `report` | بازسازی جدول‌ها و شکل‌ها از CSV/NPZ | آموزش یا تنظیم مدل ندارد |
| `all` | scan، final، calibration، ablation، matched-K، K-study، sparsity و چگالی مصنوعی | مراحل تفکیک‌شده با provenance |

`main` نتیجهٔ اسکنِ یک پوشهٔ دیگر را خودکار وارد نمی‌کند. برای استفاده از برنده‌های اسکن، `final` را در **همان output** و با همان تنظیمات اجرا کنید. در ۴٫۲، ابلیشن، K-study، sparsity و convergence نیز به‌طور پیش‌فرض از برندهٔ همان split شروع می‌شوند؛ `--analysis-source defaults` فقط برای اجرای صریحِ مطالعهٔ تنظیمات پایه است.

## مسیر کوتاه با launcher آماده

فایل `paper_v4_2.py` همان دستورهای این راهنما را بدون تکرار آرگومان‌های طولانی می‌سازد. preset پیش‌فرض آن سه split، سه seed و Kهای ۱۰،۲۰،۵۰ است؛ تعداد fitها را ابتدا در `plan` ببینید. همهٔ مراحل را با همان آرگومان‌ها نگه دارید:

```bash
python paper_v4_2.py --phase plan --dataset ml-100k --output ./paper_v4_2
python paper_v4_2.py --phase smoke --dataset ml-100k --output ./paper_v4_2 --device cuda:0
python paper_v4_2.py --phase scan --dataset ml-100k --output ./paper_v4_2 --device cuda:0
python paper_v4_2.py --phase final --dataset ml-100k --output ./paper_v4_2 --device cuda:0
python paper_v4_2.py --phase all --dataset ml-100k --output ./paper_v4_2 --device cuda:0
python paper_v4_2.py --phase report --dataset ml-100k --output ./paper_v4_2
```

`smoke` در زیرپوشهٔ جدا اجرا می‌شود. `scan` این launcher عمداً validation-only است تا فرصت بررسی پیش از TEST داشته باشید؛ این انتخاب wrapper است و با پیش‌فرض خودِ `run_v4.py scan` که final را هم ادامه می‌دهد فرق دارد. `all` مراحل کامل‌شده را با resume بازاستفاده می‌کند و گزینهٔ `--include-convergence` را نیز می‌دهد. برای دیدن فرمان دقیق بدون اجرا، `--dry-run` اضافه کنید. در ویندوز نیز از همین launcher استفاده کنید؛ مسیرها به‌صورت آرایهٔ آرگومان منتقل می‌شوند و shell interpolation ندارند.

## ۱. نصب و سلامت اولیه

در کپیِ نسخهٔ ۴، ابتدا فایل‌ها را بررسی کنید:

```bash
python verify_v4.py --require-uncommitted-base
python -m pip install -r requirements_v4.txt
python -m pytest tests -q
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

اگر بعداً خودتان روی نسخهٔ ۴ commit کردید، گزینهٔ `--require-uncommitted-base` را حذف کنید؛ بررسیِ ۶۶ فایل محافظت‌شده همچنان قابل استفاده است. وجود CUDA فقط به کارت گرافیک وابسته نیست؛ نصب PyTorch هم باید ساخت سازگار با CUDA باشد.

برای آزمون واقعی ذخیرهٔ گراف روی همان دستگاه:

```bash
python experiments/v4_execution_parity.py --device cuda:0 --steps 3 --threads 2 --output ./gpu_parity_check --cache-dir ./v4_ssd_cache
```

این دستور روی دادهٔ مصنوعی و ماسک ثابت، وزن‌ها **و وضعیت AdamW** را برای `resident/auto/host/ssd` مقایسه و `EXECUTION_PARITY.json` تولید می‌کند. برای readoutهای تازه، همان دستور را با `--readout coupled` و `--readout decoder` و outputهای جدا نیز اجرا کنید. به هیچ split واقعیِ مقاله دست نمی‌زند. نتیجهٔ CPU جای تأیید CUDA نیست. برابری این آزمون نیز اثبات عملکرد تمام اندازه‌های داده و تمام کرنل‌ها نیست.

## ۲. smoke برای بررسی خط لوله

```bash
python run_v4.py scan --datasets ml-100k --data-root ./datasets --output ./results_v4_2_smoke --methods MF,AutoRec,Method4 --k 20 --split-seeds 2026 --model-seeds 1 --quick --v4-search-jobs-per-method 1 --include-blend --v4-device cuda:0 --v4-storage auto --v4-cache-dir ./v4_ssd_cache
```

`--quick` عمداً بودجه، تعداد seedها و برخی محاسبات داخلی را کوچک می‌کند. نتیجهٔ آن برای مقاله نیست و شکل‌هایش با نشانهٔ SMOKE مشخص می‌شوند. محدودیت `--v4-search-jobs-per-method` نیز جست‌وجوی ناقص است؛ آن را در اجرای مقاله حذف کنید. smoke فقط سالم‌بودن مسیر داده، آموزش، انتخاب، checkpoint، خروجی و رسم را نشان می‌دهد.

## ۳. پیش از هزینهٔ زیاد، طرح کامل را ببینید

نمونهٔ مطالعهٔ MovieLens-100K؛ تمام فرمان‌های بعدی باید همین مجموعهٔ آرگومان‌ها را نگه دارند:

```bash
python run_v4.py plan --datasets ml-100k --data-root ./datasets --output ./paper_v4_2_ml100k --k 5,10,20,50,64 --split-seeds 2026,2027,2028,2029,2030 --model-seeds 11,22,33,44,55 --scan-mode hybrid --v4-config examples/v4/recommended_v4_2_ml100k.json
```

طرح بزرگِ نمونه حدود ۲۸ هزار fit کاندیدا دارد و هزینهٔ OOF هم به آن اضافه می‌شود؛ قبل از اجرا خروجی واقعی `plan` را بخوانید. این تعداد split/seed شرط کافیِ استقلال آماری نیست. برای توسعه، ابتدا یک split/seed و grid از پیش محدودشده را فقط روی ولیدیشن اجرا کنید؛ طرح کامل مقاله باید با بودجهٔ واقعی دستگاه سازگار باشد. تقسیم‌های تصادفیِ یک دیتاست ممکن است همپوشانی داشته باشند. داده، seed، grid و قاعدهٔ انتخاب را پیش از مشاهدهٔ TEST ثبت کنید. تعداد خام آموزش‌ها در `plan` دیده می‌شود، اما هزینهٔ OOF روش ۲، ensemble روش ۷ و solverها با یک fit ساده برابر نیست؛ گزارش هزینهٔ واقعی را هم بررسی کنید.

تنظیمات `recommended_v4_2_ml100k.json` از مطالعهٔ اکتشافیِ همین MovieLens-100K آمده‌اند. آن را به‌عنوان optimum تمام دیتاست‌ها معرفی نکنید. در دیتاست‌های دیگر، از grid مستقل همان دیتاست استفاده کنید. نسخه‌های قدیمیِ TEST این پروژه قبلاً دیده شده‌اند؛ آن‌ها holdout تأییدیِ تازه نیستند.

## ۴. جست‌وجو، همراه همهٔ داده‌های لازم، اما فعلاً بدون TEST

```bash
python run_v4.py scan --scan-only --datasets ml-100k --data-root ./datasets --output ./paper_v4_2_ml100k --k 5,10,20,50,64 --split-seeds 2026,2027,2028,2029,2030 --model-seeds 11,22,33,44,55 --scan-mode hybrid --v4-config examples/v4/recommended_v4_2_ml100k.json --include-blend --v4-device cuda:0 --v4-storage auto --v4-host-budget-mb 2048 --v4-cache-dir ./v4_ssd_cache
```

در هر **outer split جداگانه**، میانگین RMSE ولیدیشن روی model-seedهای یکسان کمینه می‌شود. یک تنظیم واحد با تجمیع تمام splitها انتخاب نمی‌شود؛ زیرا ولیدیشن split دیگر ممکن است شامل TEST این split باشد. tie-break نیز در manifest ثبت می‌شود. بهترین checkpoint هر fit با selection validation انتخاب می‌شود. TEST در جدول انتخاب وجود ندارد.

برای روش ۷، ولیدیشن به سه نقشِ مجزا تقسیم می‌شود: selection، probability-fit و probability-hold. همهٔ مدل‌ها همان selection را برای خطای نقطه‌ای دارند. مرحلهٔ probability فقط شکل توزیع را یاد می‌گیرد و تغییر میانگین با آزمون بررسی می‌شود. مصرف آن دو بخش اضافه برای برازش توزیع، در گزارش صریح است.

برای سبک‌کردن ذخیرهٔ وزن‌های موقت می‌توانید `--v4-retain-candidates metrics` اضافه کنید. برای رسم در پایان به‌جای زمان آموزش می‌توانید `--v4-no-plots` اضافه کنید و بعداً `report` بزنید. این دو گزینه را هنگام `--resume` مطابق اجرای اولیه نگه دارید.

## ۵. ارتباط اسکن و آزمایش اصلی

پس از بررسی ولیدیشن و تثبیت طرح:

```bash
python run_v4.py final --datasets ml-100k --data-root ./datasets --output ./paper_v4_2_ml100k --k 5,10,20,50,64 --split-seeds 2026,2027,2028,2029,2030 --model-seeds 11,22,33,44,55 --scan-mode hybrid --v4-config examples/v4/recommended_v4_2_ml100k.json --include-blend --v4-device cuda:0 --v4-storage auto --v4-host-budget-mb 2048 --v4-cache-dir ./v4_ssd_cache
```

`final` از `search/validation_winners.csv`، `config.json`های اسکن و امضای داده استفاده می‌کند. اگر تنظیم علمی، داده یا سورس تغییر کرده باشد، کار متوقف می‌شود. هر split همان برندهٔ خودش، K و initialization منتخب را می‌گیرد. دادهٔ ولیدیشن به TRAIN اضافه نمی‌شود؛ نقش‌ها و قاعدهٔ انتخاب checkpoint حفظ می‌شوند. مدل‌های منتخب، پیش‌بینی‌های TRAIN/VAL/TEST و گزارش نهایی ذخیره می‌شوند.

اگر می‌خواهید اسکن و final یک‌جا اجرا شوند، همان دستور بخش ۴ را بدون `--scan-only` اجرا کنید. برای ادامهٔ کار قطع‌شده، `--resume` را با همان تنظیمات اضافه کنید. resume آموزشِ job کامل‌شده را تکرار نمی‌کند و گزارشش را بازسازی می‌کند. برای ادامهٔ نسخهٔ ۴٫۱ از output تازه استفاده کنید؛ تغییر سورس در ۴٫۲ نباید به‌عنوان ادامهٔ همان اجرای قبلی جا زده شود.

## ۶. کل مجموعهٔ آزمایش‌های مقاله

یک فرمان برای تمام مراحلِ پیش‌فرض، با استفاده از کارهای کامل‌شده:

```bash
python run_v4.py all --resume --datasets ml-100k --data-root ./datasets --output ./paper_v4_2_ml100k --k 5,10,20,50,64 --split-seeds 2026,2027,2028,2029,2030 --model-seeds 11,22,33,44,55 --scan-mode hybrid --v4-config examples/v4/recommended_v4_2_ml100k.json --analysis-source winners --ablation-fixed-k 20 --keep-rates 0.3,0.5,0.7,1.0 --include-blend --include-convergence --v4-device cuda:0 --v4-storage auto --v4-host-budget-mb 2048 --v4-cache-dir ./v4_ssd_cache
```

این فرمان بسیار پرهزینه‌تر از یک `main` است. `all` ابتدا اسکنِ کامل‌شده و final را با resume بازمی‌خواند، سپس مطالعه‌های دیگر را اجرا می‌کند. `calibration` فقط وقتی Method7 در مطالعه باشد معنا دارد. آزمایش چگالی تولیدشونده فقط برای synthetic اجرا می‌شود. دادهٔ per-pair و TRAIN هر کاندیدا عمداً نگه داشته می‌شود؛ حجم دیسک با تعداد fit و تعامل‌ها رشد می‌کند و حذف وزن‌های موقت به‌تنهایی این حجم را حذف نمی‌کند.

برای اجرای جداگانه، فقط کلمهٔ ابتدای دستور بالا را با `ablation`، `ablation-fixed`، `kstudy`، `sparsity`، `calibration` یا `convergence` عوض کنید و همان تنظیمات مطالعه را نگه دارید. `convergence` validation-only است و زمان‌بندی انتخاب checkpoint برنده را عوض نمی‌کند. نمودارهای عادیِ train/val نیز بدون این بازاجرای اضافه، در گزارش اصلی وجود دارند.

برای مطالعهٔ زمانی MovieLens، `--include-temporal` را به `all` اضافه کنید. این مسیر اسکن و انتخاب مستقل خودش را روی تقسیم زمانی انجام می‌دهد؛ تنظیمِ تقسیم تصادفی بدون بررسی به آن منتقل نمی‌شود.

معنای ابلیشن‌ها:

- `full` دقیقاً از برندهٔ اسکن همان split آغاز می‌شود؛ فقط مکانیسم اعلام‌شده در نسخهٔ حذف‌شده تغییر می‌کند.
- اگر حذف یک بازنمایی، readout وابسته به آن را نیز ناممکن کند، آن مسیر وابسته حذف و تغییر کامل config ثبت می‌شود؛ آن را حذفِ یک ضریب مستقل تعبیر نکنید.
- K، بودجهٔ اعلام‌شده و تنظیمات دیگر ثابت‌اند، اما حذف عضو ensemble می‌تواند تعداد پارامترها و توقف زودهنگام می‌تواند هزینهٔ واقعی را عوض کند. ستون‌های واقعی را گزارش کنید.
- در `ablation-fixed`، K برای full و تمام حذف‌ها یکسان است. این تحلیل با full منتخبِ K دیگری مخلوط نمی‌شود.
- در `sparsity` فقط TRAIN تنک می‌شود. نگهداری حداقل پشتیبانی کاربر/آیتم ممکن است تعداد یال‌ها را از نسبت درخواستی بیشتر نگه دارد؛ فایل split تعداد واقعی را نشان می‌دهد.
- نمودار TEST در K-study یا sparsity توصیفی است؛ بازگشت از آن به تنظیم مدل، TEST را وارد چرخهٔ توسعه می‌کند.

## ۷. فایل‌ها و شکل‌ها دقیقاً کجا هستند؟

`ROOT` در این جدول همان `paper_v4_2_ml100k/ml-100k` است.

| مسیر | کاربرد |
|---|---|
| `search/hyperparameter_validation_all.csv` | تمام تلاش‌های جست‌وجو، بدون ستون TEST |
| `search/validation_winners.csv` | تنظیم منتخب هر روش و split |
| `search/validation_ranked_configs.csv` | رتبهٔ کامل کاندیداهای ولیدیشن |
| `search/validation_selection_manifest.json` | قاعده و provenance انتخاب |
| `search/publication/tables/selection_trace.csv` | علامت برنده و ترتیب انتخاب |
| `search/publication/tables/search_budget.csv` | تعداد واقعی تلاش‌ها و زمان هر خانواده |
| `search/publication/tables/hyperparameters_long.csv` | نام/مقدار پارامترها در قالب long |
| `search/publication/tables/hyperparameter_sensitivity.csv` | دادهٔ نمودار حساسیت؛ هر نقطه یک config کامل است |
| `search/publication/figures/` | شکل‌های برنده‌های ولیدیشن و فرایند/حساسیت جست‌وجو |
| `search/jobs/JOB/` | تمام داده و گزارش هر job جست‌وجو |
| `final/METHOD/split_SEED/selected_models/` | وزن قابل بازخوانیِ برنده‌ها |
| `final_locked_test_all.csv` | آزمایش اصلی برای همهٔ روش‌ها |
| `final_locked_test_summary.csv` | خلاصهٔ آزمایش اصلی |
| `publication/tables/per_pair/*.csv.gz` | زوج کاربر–آیتم، truth، prediction، residual و تشخیص‌های موجود |
| `publication/tables/conditional_errors.csv` | خطا به تفکیک امتیاز و تعداد مشاهدهٔ کاربر/آیتم |
| `publication/tables/recomputed_prediction_metrics.csv` | محاسبهٔ مستقل از NPZ؛ خام و با کلیپ یکسان در صورت وجود دامنهٔ داده |
| `publication/tables/point_prediction_calibration.csv` | میانگین واقعی در برابر میانگین پیش‌بینی‌شده در binها |
| `publication/tables/epoch_history.csv` | تاریخچهٔ آموزش و ولیدیشن |
| `publication/tables/training_budget.csv` | بهترین epoch، سقف بودجه، توقف و شمار کارهای ثبت‌شده |
| `publication/tables/phase_and_execution_diagnostics.csv` | OOF، برازش احتمال و جزئیات اجرای تانسورها |
| `publication/tables/device_utilization_samples.csv` | نمونه‌های NVML، در صورت دسترس‌بودن CUDA/NVML |
| `publication/tables/absolute_error_ecdf.csv` | دادهٔ ECDF خطای مطلق |
| `publication/tables/residual_histogram.csv` | هیستوگرام خطا |
| `publication/tables/probability_coverage.csv` | پوشش و اندازهٔ بازهٔ احتمالی واقعیِ روش ۷ |
| `publication/tables/probability/` | reliability و randomized PIT در حالت ordinal |
| `publication/figures/` | رتبهٔ خطا، train/val، gap، دُم خطا، فعالیت، ranking، احتمال و هزینه |
| `publication/REPORT_MANIFEST.json` | منابع شکل/جدول، hash، smoke و معنای هر شاخص |
| `mechanisms/*/training_steps.csv` | lossها، ماسک، تعداد هدف و norm گرادیان در هر گام |
| `ablations/` و `ablations_matched_K_20/` | ابلیشنِ منتخب و matched-K، همراه config هر حذف |
| `kstudy/` و `sparsity/` | جدول خام و دادهٔ حساسیت |
| `search/ensemble/VALIDATION_WEIGHTS.json` | وزن کنترل ترکیبی، ثابت‌شده پیش از خواندن TEST |
| `ensemble/` | پیش‌بینی و نتیجهٔ کنترل ترکیبی؛ جدا از هفت مدل |

جدول‌های خالیِ یک نوع خروجی، schema مشخص دارند؛ خالی‌بودن پوشش احتمالیِ یک مدل نقطه‌ای خطا نیست. شکل‌ها فقط برای شاخص‌ها و داده‌های موجود ساخته می‌شوند. مثلاً MF توزیع احتمالی ندارد که نمودار NLL یا PIT واقعی از آن بسازیم. `NDCG_testset` رتبه‌بندی بین آیتم‌های امتیازخوردهٔ held-out است، نه retrieval کل کاتالوگ؛ نام و کپشن آن را در مقاله حفظ کنید.

## ۸. بازسازی نمودارهای شیک بدون آموزش مجدد

```bash
python run_v4.py report --datasets ml-100k --output ./paper_v4_2_ml100k
```

شکل‌های گزارش جدیدِ `publication/` و نمودارهای V4 در PNG با ۴۰۰ DPI، PDF برداری و SVG قابل ویرایش ذخیره می‌شوند. نمودارهای اضافیِ قدیمی با پیشوند `q1_` از هستهٔ ثابت، همان PNG/PDF پیشین را دارند؛ برای مقاله از خروجی جدید استفاده کنید. رنگ‌ها میان مدل‌ها ثابت، فونت PDF قابل embed و متن SVG قابل ویرایش است. محورهای خامِ امتیاز، تعداد داده و نوع uncertainty مشخص‌اند. یک split، error bar ساختگی ندارد؛ چند split، پراکندگی بین splitها را نشان می‌دهد، نه خودبه‌خود CI یا تضمین استقلال. دادهٔ CSV هر نمودار باقی می‌ماند تا بتوانید اندازهٔ شکل و سبک نهایی مجله را عوض کنید.

## ۹. GPU، حافظه و SSD با حفظ تعریف آموزش

پیکربندی پیشنهادی پایه برای یک GPU:

```text
--v4-device cuda:0 --v4-storage auto --v4-memory-reserve 0.15 --v4-host-budget-mb 2048 --v4-cache-dir ./v4_ssd_cache
```

بودجهٔ RAM را متناسب با RAM آزاد دستگاه خود انتخاب کنید؛ مقدار پیش‌فرض ۱۰۲۴ MiB است. `auto` بر اساس حافظهٔ آزاد CUDA، saved tensorهای autograd را resident نگه می‌دارد یا به RAM/SSD می‌برد. نسخهٔ ۴٫۲ وزن‌ها/بافرهایی را که از قبل زنده‌اند بیهوده offload نمی‌کند و ذخیره‌های تکراریِ معتبر را با کنترل هویت و نسخهٔ tensor بازاستفاده می‌کند. حافظهٔ pinned برای انتقال RAM→GPU در صورت مناسب‌بودن استفاده می‌شود.

در OOMِ CUDA، کل کاندیدا از حالت اولیه و RNG اولیه بازاجرا می‌شود: ابتدا auto، سپس host و در نهایت SSD. optimizer نیمه‌آموزش‌دیده ادامه داده نمی‌شود. dtype، batch منطقی، ماسک‌ها، تعداد updateها و نرخ یادگیری عوض نمی‌شوند. فایل موقت گراف پس از استفاده پاک می‌شود. SSD راه افزایش ظرفیت حافظه است؛ در حالت عادی می‌تواند کندتر از resident باشد.

**Overflow عددی با OOM فرق دارد.** برای حفظ شرط شما، precision، LR و batch به‌طور خاموش عوض نمی‌شوند. loss/gradient غیرمتناهی باعث شکست صریحِ job پیش از معرفی نتیجهٔ معتبر می‌شود. رفع چنین مسئله‌ای نیازمند اصلاح عددیِ مشخص و آزمون‌شده است؛ کاهش خودکار LR یا فعال‌کردن AMP یک تغییر در آموزش محسوب می‌شود.

پرشدن درصد بیشتری از VRAM، معادل استفادهٔ بهتر از GPU نیست. مدل کوچک، gather/scatter تنک، وابستگی‌های solver و انتقال داده می‌توانند محدودکننده باشند. `device_kernel_active_percent` و فعالیت گذرگاه حافظه، سنجه‌های device-wide از NVML هستند؛ FLOPs واقعی یا سهم فقط این پردازش نیستند. زمان stream، زمان کل، حافظهٔ peak و مقدار offload جدا ثبت می‌شوند. این محیط CUDA واقعی نداشت؛ تضمین سرعت یا برابری bitwise روی GPU شما از نتایج CPU نتیجه نمی‌شود.

## ۱۰. انصاف رقبا و گزارش مقاله

- AutoRec با regularization ضعیف، معیار کافی برای مقاله نیست. grid اصلیِ regularization و انتخاب optimizer ثبت شده است؛ اسکن ML-100K کنترل‌های نزدیک به ظرفیتِ روش‌های۴/۶ با H=146,265 و L2=10,100 را نیز دارد. AutoRec تنظیم‌شده را حذف نکنید.
- برای MBMF، optimizer اصلی مقاله و کنترل Adam در پروتکل common تفکیک می‌شوند. هر دو همان عوامل، شعاع‌ها، target و mask را دارند، اما مسیر بهینه‌سازی یکسان نیست؛ Adam را بازتولید Algorithm 1 معرفی نکنید.
- MF/BiasMF و AEMC-NE نیز آخرین epoch را حتی اگر مضرب `eval_every` نباشد ارزیابی می‌کنند. «بهترین epoch در سقف» به‌معنای همگرایی نیست.
- نتایج AEMC-NE این پروژه بازپیاده‌سازی معادلات‌اند. عدد ضعیف این کد به‌تنهایی مجوز ادعای شکست مقالهٔ اصلی نیست. پروتکل مقالهٔ AEMC-NE حدود ۸۵٫۵٪ TRAIN / ۴٫۵٪ VAL / ۱۰٪ TEST و AutoRec حدود ۸۱٪ / ۹٪ / ۱۰٪ است؛ با پروتکل common مخلوط نشوند.
- نسخهٔ پذیرفته‌شدهٔ MBMF که از مخزن نویسنده خوانده شد، MovieLens را در جدول دیتاست‌های آزمایشی ندارد. حتی در جدول‌های خود مقاله نیز MF در بعضی Kها از MBMF-F بهتر است. انتظار برتری همیشگیِ MBMF درست نیست.
- پارامتر، تعداد هدف، تلاش‌های تنظیم، OOF، ensemble و زمان واقعی را گزارش کنید. «هر دو K=20» یعنی ابعاد latent برابر است؛ مساوی‌بودن ظرفیت و هزینه از آن نتیجه نمی‌شود.
- یک کنترل ترکیبیِ ساده نیز در این نسخه هست. اگر ensemble ساده دقت بهتری دارد، اضافه‌کردن هندسه را صرفاً با نمودار RMSE توجیه نکنید؛ مزیتِ متمایز احتمالی/پایداری باید جدا سنجیده شود.

راهنمای منابع و عددهای واقعی این نوبت در `V4_2_REVIEW_FA.md` و مرز ادعاهای ریاضی/نوآوری در `V4_2_THEORY_FA.md` آمده‌اند.

منابع اجرایی: [Saved tensor hooks در PyTorch](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.saved_tensors_hooks)، [CUDA memory](https://docs.pytorch.org/docs/2.14/notes/cuda.html)، [معنای utilization](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.utilization.html) و [معنای memory_usage](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory_usage.html).
