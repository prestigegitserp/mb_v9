> به‌روزرسانی 4.2: [راهنمای کامل آزمایش‌های مقاله](docs/V4_2_PAPER_WORKFLOW_FA.md)، [گزارش دیباگ و شواهد](docs/V4_2_REVIEW_FA.md) و [ممیزی نظری](docs/V4_2_THEORY_FA.md). نسخه‌های 4.0 و 4.1 در بخش‌های بعدی فقط سابقه‌اند.

> به‌روزرسانی 4.1: [گزارش بازبینی و نتایج تازه](docs/V4_1_REVIEW_FA.md) و [منطق ریاضی](docs/V4_1_THEORY_FA.md). تنظیمات منتخب: `examples/v4/reviewed_ml100k.json`. بخش‌های زیر سابقهٔ بستهٔ اولیهٔ V4.0 هستند.

# نسخهٔ چهارم: شاخهٔ پژوهشیِ بدون کامیت

V4 هفت مدل را در `v4/` جایگزین می‌کند. `run_v2.py`، `run_v3.py`، تمام کدهای `v2/`، `v3/`، `v9/` و `vendor_v8/` و مستندات قبلی عیناً محفوظ‌اند. فقط `VERSION` و dispatcher مشترک `run_v9.py` از فایل‌های موجود تغییر کرده‌اند. پایهٔ تحویل V3.0.1 با SHA `3b1af9306a9ef9da541c5b0fabbe74dbafeb3011` است. هیچ commit، merge یا push جدیدی در این تحویل انجام نشده است.

برای نصب وابستگی‌های اجرا و telemetry اختیاری GPU: `pip install -r requirements_v4.txt`.

## اجرای همان پایلوت آشنا

دستور زیر بودجهٔ کامل پیش‌فرض را اجرا می‌کند؛ `--quick` ندارد:

```bash
python run_v9.py main --dataset ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v4/laptop_pilot
```

`main` تنظیمات پیش‌فرض را مقایسه می‌کند و اسکن هایپرپارامتر نیست. پیش‌فرض‌ها «فرضیهٔ پژوهشی» هستند؛ خصوصاً برتری نسبت به رقبای تنظیم‌شده از این اجرا نتیجه نمی‌شود. ورودی صریح V4 هم `python run_v4.py ...` است. برای نسخه‌های قبلی از entry مستقل یا `python run_v9.py --code-version 3 ...` و `--code-version 2` استفاده کنید.

آزمون سریعِ سلامت، با کیفیت نامناسب برای مقاله:

```bash
python run_v4.py main --dataset ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --quick --output ./results_v4/smoke
```

## اسکن واقعاً یک مطالعهٔ کامل است

```bash
python run_v4.py scan --dataset ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --scan-mode hybrid --output ./results_v4/tuned_pilot
```

این دستور به ترتیب همهٔ candidateها را فقط روی train/validation می‌سنجد، تنظیم برنده را برای هر مدل و split با میانگین seedهای مدل قفل می‌کند، برنده را با همان داده و تنظیم دوباره آموزش می‌دهد و سپس ارزیابی test و گزارش نهایی را می‌سازد. انتخاب epoch درون هر آموزش هم فقط روی بخش selection ولیدیشن است. پیش‌بینی‌های candidate، checkpoint، ریزدادهٔ خطا، تاریخچهٔ آموزش و شکل‌های مجاز آن مرحله ذخیره می‌شوند. `--scan-only` توقف عمدی پس از جست‌وجو است؛ در آن حالت نبودِ نمودار test صحیح است. `all` افزون بر این مسیر، ablation، حساسیت K، sparsity و density و مطالعهٔ جداگانهٔ کالیبراسیون را اجرا می‌کند؛ `scan` این آزمایش‌های اضافی را خودکار به‌جای جست‌وجوی هایپرپارامتر انجام نمی‌دهد.

جست‌وجوی کامل ارزان نیست. محدودکردن عمدی با `--v4-search-jobs-per-method 3` صرفاً یک جست‌وجوی محدود است و ممکن است به مقادیر مهم مانند λ=100 نرسد. مقدار پیش‌فرض صفر یعنی بدون سقف. شبکهٔ AutoRec شامل 0.001، 0.01، 0.1، 1، 100، 1000 است؛ برای AEMC-NE دو ضریب مستقل در شبکهٔ مقاله نیز بررسی می‌شوند. محدودیت بودجه را در گزارش مقاله اعلام کنید.

ادامهٔ کار با همان تنظیمات و همان سورس:

```bash
python run_v4.py scan --dataset ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --scan-mode hybrid --resume --output ./results_v4/tuned_pilot
```

تغییر سورس، داده یا تنظیم قفل‌شده با همان خروجی پذیرفته نمی‌شود؛ خروجی تازه انتخاب کنید. `resume` کارهای کامل را بازآموزی نمی‌کند؛ بازسازی نمودارها مجاز است. بازاجرای گزارش بدون آموزش:

```bash
python run_v4.py report --dataset ml-100k --output ./results_v4/tuned_pilot
```

## خروجی‌ها

| مسیر زیر پوشهٔ دیتاست | محتوا |
|---|---|
| `main/publication/` | گزارش اجرای پیش‌فرض |
| `search/jobs/<job>/publication/` | تمام جدول‌ها و شکل‌های train/validation همان candidate؛ بدون test |
| `search/validation_winners.csv` | تنظیم برنده برای هر مدل و split |
| `final/<method>/split_<seed>/` | آموزش و ارزیابی قفل‌شده، checkpoint و پیش‌بینی‌ها |
| `publication/` | گزارش ترکیبی نهاییِ scan |
| `publication/tables/per_pair/*.csv.gz` | truth، prediction، residual، خطا و اطلاعات ساختاری هر تعامل |
| `publication/tables/` | جدول‌های بلند معیارها، میانگین split، پراکندگی seed، خطای مشروط، ECDF، مقایسهٔ جفت‌شده با MF و هزینه |
| `publication/tables/probability/` | قابلیت اعتماد CDF و PIT با seed تشخیصی ثابت برای M7 ترتیبی |
| `publication/figures/` | PDF و SVG برداری و PNG با 400dpi |
| `mechanisms/` | loss، norm گرادیان و تعداد target در هر گام؛ OOF و تاریخچهٔ آن برای M2 |
| `predictions/calibration/` | خروجی جداگانهٔ probability fit/hold برای M7 |
| `V4_ENVIRONMENT.json` | هش پیاده‌سازی، نسخهٔ بسته‌ها، precision، CUDA و سیاست حافظه |
| `publication/REPORT_MANIFEST.json` | منشأ و SHA فایل‌های داده و محدودیت تفسیر شکل‌ها |

نمودارها شامل رتبه‌بندی معیارهای اصلی، train/validation در طول آموزش، هدف‌های router/shape، tail error، خطای گروه‌های rating و degree، هزینه/دقت، حدود ساختاری و تشخیص احتمالاتی هستند؛ تنها وقتی معیار مربوط واقعاً موجود باشد تولید می‌شوند. شکل‌های smoke واترمارک دارند. برای یک split بازهٔ اطمینان بین splitها ساخته نمی‌شود؛ در چند split، seedهای مدل ابتدا درون split میانگین می‌شوند و میله/نوار SD بین splitهاست. `NDCG_testset` رتبه‌بندی بین آیتم‌های امتیازخوردهٔ held-out است. حدود ساختاری، support پیش‌بینی نقطه‌ای‌اند و interval احتمالاتی نیستند.

## CUDA، RAM و SSD بدون تغییر مسیر علمی

```bash
python run_v4.py main --dataset ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --v4-storage auto --v4-host-budget-mb 2048 --v4-cache-dir ./ssd_cache --output ./results_v4/gpu_pilot
```

در صورت وجود CUDA، `auto` ابتدا resident است؛ saved tensorهای autograd هنگام فشار حافظه بدون کاهش dtype به RAM و پس از بودجهٔ RAM به SSD منتقل می‌شوند. در CUDA OOM، کل candidate از state اولیه و RNG اولیه با سیاست حافظهٔ بعدی بازپخش می‌شود؛ optimizer نیمه‌اجرا ادامه داده نمی‌شود. mask، batch منطقی، شمار optimizer step، probe، precision و learning rate عوض نمی‌شوند. فایل‌های موقت گراف پاک می‌شوند. `resident`، `host` و `ssd` هم صریحاً قابل انتخاب‌اند. دادهٔ ورودی و پارامترهایی که الزاماً resident هستند با این روش خارج نمی‌شوند؛ بنابراین تضمین اجرای هر اندازهٔ دیتاست وجود ندارد.

Overflow عددی به تغییر پنهانی مدل یا precision منجر نمی‌شود؛ با خطای روشن متوقف می‌شود. تضمین تغییرندادن آموزش با کاهش خودکار learning rate یا AMP سازگار نیست. `--v4-retain-candidates metrics` وزن‌های موقت candidate را حذف می‌کند، اما پیش‌بینی‌ها، trace و دادهٔ گزارش را نگه می‌دارد؛ وزن‌های منتخب مرحلهٔ final ذخیره می‌شوند. `--v4-no-plots` تنها رندر شکل‌ها را خاموش می‌کند. `--no-epoch-train-rmse` ارزیابی اضافهٔ train در هر تیک را حذف می‌کند؛ به optimization دست نمی‌زند.

برابری state مدل و Adam در resident/host/SSD روی CPU برای هر هفت مدل آزموده شده است. CUDA در محیط تحویل در دسترس نبود؛ سرعت، حداکثر utilization و برابری بیتی CUDA تأیید نشده‌اند. NVML درصد زمان فعال‌بودن device و memory bus را در صورت دسترسی ثبت می‌کند؛ این‌ها درصد FLOPs یا استفادهٔ اختصاصی همین process نیستند. نوشتن روی SSD الزاماً سرعت را زیاد نمی‌کند.

## مقایسه با مقاله و اعتبار علمی

پروتکل مشترک برای همهٔ مدل‌ها، validation اصلی را به selection، probability-fit و probability-hold تقسیم می‌کند و train/test اصلی را نگه می‌دارد. حالت‌های جداگانهٔ `--v4-protocol aemc-paper` و `autorec-paper` split سراسری با نسبت‌های مؤثر 85.5/4.5/10 و 81/9/10 می‌سازند؛ برای این حالت‌ها M7 را صریحاً کنار بگذارید چون نقش‌های کالیبراسیون مستقل ندارد. این گزینه‌ها بازتولید کامل کد نویسنده یا تضمین عدد مقاله نیستند. نتیجهٔ پروتکل‌های متفاوت را در یک جدول اصلی بدون برچسب مخلوط نکنید.

جزئیات نظری در `V4_DESIGN.md`، ممیزی و نتایج تازه در `docs/V4_AUDIT_FA.md` و شواهد ماشینی در `V4_VERIFICATION.json` هستند. نوآوری کامل یا بهترشدن همهٔ هفت مدل تضمین نشده است. کنترل‌های ساده‌تر و ablation برای هر مکانیسم در کد وجود دارند؛ برای انتساب پیشرفت به خود ایده، باید در ظرفیت و بودجهٔ آموزش هم تطبیق انجام شود.

```bash
python verify_v4.py --require-uncommitted-base
python -m pytest tests -q -p no:cacheprovider
```
