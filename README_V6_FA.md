# V6 — شاخهٔ توسعه، بدون تغییر نسخه‌های منتشرشده

این نسخه روی شاخهٔ v6-development توسعه می‌یابد. فایل VERSION و ورودی‌های V1 تا V5 تغییر نکرده‌اند. برای اجرای V6 باید صریحاً run_v6.py یا paper_v6.py را انتخاب کنید. شاخهٔ انتشار v6 ایجاد یا merge نشده است.

**این توسعه، ادعای رسیدن به RMSE مقاله یا پایین‌تر از 0.87 نیست.** نتیجهٔ تازه باید از اجرای واقعی و هم‌پروتکل به دست بیاید. آزمون‌های نرم‌افزاری و اجرای محدود validation چنین ادعایی را ثابت نمی‌کنند.

## شروع در پوشهٔ جدا

~~~powershell
git clone --branch v6-development --single-branch https://github.com/prestigegitserp/mb_v9.git mb_v9_v6_dev
cd mb_v9_v6_dev
python -m pip install -r requirements_v6.txt
python verify_v6.py
python paper_v6.py --stage plan --datasets ml-100k --output ./results_v6/develop
python paper_v6.py --stage develop --datasets ml-100k --data-root ./datasets --output ./results_v6/develop --device cuda
~~~

develop جست‌وجوی محدود، انتخاب با validation، ارزیابی منتخب‌ها روی test و گزارش‌های نهایی را انجام می‌دهد. این حالت smoke نیست و بودجهٔ هر مدل را به سه epoch محدود نمی‌کند. پیش‌فرض یک split، یک model seed، K=20 و حداکثر هشت job برای هر روش است. به تعداد آموزش‌هایی که plan چاپ می‌کند توجه کنید؛ بعضی رقبا سه initialization دارند.

برای بررسی صرفاً نرم‌افزاری و کوتاه، به develop گزینهٔ --quick و یک مسیر خروجی جدا بدهید. نتایج quick برای مقاله مناسب نیستند.

برای ادامهٔ اجرای قطع‌شده همان دستور را دوباره اجرا کنید؛ launcher گزینهٔ resume را اضافه می‌کند. job کامل از نو آموزش نمی‌بیند. اگر فقط تولید گزارش شکست خورده باشد، وضعیت REPORT_FAILED با training_complete نگهداری می‌شود و resume صرفاً گزارش را دوباره می‌سازد. آموزش نیمه‌کاره، epoch-resume ندارد و از ابتدای همان job شروع می‌شود.

## قالب‌ها و آزمایش‌های مقاله

در examples/v6 برای هر شش دیتاست قالب مستقل وجود دارد:

| پوشه | کاربرد |
|---|---|
| development | گزینه‌های متنوع و محدود برای توسعه |
| paper | فضای جست‌وجوی جامعِ تعریف‌شده، با سقف job برابر صفر |
| v5_compatible | JSON قابل‌خواندن با run_v5_2.py، بدون تغییر سورس V5 |

هیچ قالبی تنظیم بهینهٔ اندازه‌گیری‌شدهٔ آن دیتاست معرفی نمی‌شود. پروتکل، ظرفیت و مقدارهای اولیه را باید از نتیجهٔ validation جدا دانست.

راهنمای مرحله‌به‌مرحلهٔ تمام دیتاست‌ها، ارتباط اسکن با ارزیابی اصلی و دستورهای ابلیشن در [docs/V6_EXPERIMENTS_FA.md](docs/V6_EXPERIMENTS_FA.md) آمده است. بررسی علمی تغییرات، محدودیت شواهد تاریخی، بازتولید رقبا و معادلات حافظه در [docs/V6_RESEARCH_REVIEW_FA.md](docs/V6_RESEARCH_REVIEW_FA.md) است.

[شواهد اجرا و آزمون‌ها](docs/v6_evidence/BUILD_STATUS.json)، [۲۶ آموزش validation روی ML-100K](docs/v6_evidence/ML100K_VALIDATION_PROBES.csv) و [بودجه و provenance آن‌ها](docs/v6_evidence/ML100K_VALIDATION_PROVENANCE.json) جدا ثبت شده‌اند. بهبود validation، نتیجهٔ TEST یا بازتولید قطعی مقاله معرفی نمی‌شود.

## خروجی‌ها و دیسک

در هر dataset:

- search/validation_winners.csv و search/best_configurations.json: انتخاب مستقل هر مدل در هر outer split.
- search/hyperparameter_validation_all.csv: امتیاز همهٔ تنظیم‌های اجراشده، بدون ستون test.
- search/publication/research_tables.xlsx: جدول‌های جست‌وجو، sensitivity، بودجه و انتخاب.
- final_locked_test_summary.csv: جدول نهایی test برای تنظیم‌های منتخب.
- publication/research_tables.xlsx: گزارش اصلی، تنظیم‌های منتخب، مقایسهٔ عددی با مرجع مقاله و معیارهای بازحساب‌شده.
- publication/tables/per_pair: CSV فشردهٔ کاملِ پیش‌بینی‌های منتخب؛ آرشیوهای عددی اصلی در predictions هر اجرای نهایی.
- figures و publication/figures: PNG با 400 DPI، PDF برداری و SVG.
- V6_ARTIFACT_RETENTION.json در search: وضعیت نگهداری کاندیداها.

حالت selected اطلاعات عددی و تاریخچهٔ تمام کاندیداها را نگه می‌دارد؛ آرشیو پیش‌بینی و trace حجیم فقط برای رهبران فعلی validation نگهداری می‌شود، تا جزئیات منتخب نهایی قابل گزارش بمانند. پیش‌بینی train در اسکن ذخیره نمی‌شود؛ معیارها و منحنی train محفوظ‌اند. در final، پیش‌بینی‌های کامل train/validation/test و checkpoint منتخب تولید می‌شوند. برای نگهداری تمام فایل‌های تمام trialها، artifact_policy را در یک مطالعهٔ تازه به all تغییر دهید و retain-candidates را all بدهید.

این سیاست سقف مطلق حجم گزارش نهایی نیست. نگهداری دادهٔ کامل 10M برای ده‌ها تکرار همچنان فضای قابل‌توجهی می‌خواهد. مقدار --ssd-budget-gb فقط سهمیهٔ cache موقت autograd است، نه سهمیهٔ کل نتایج.

## منابع اجرا

پیش‌فرض auto ابتدا CUDA را انتخاب می‌کند. ذخیرهٔ activation بین GPU، RAM و SSD جابه‌جا می‌شود و در OOM یا خطای عددی قابل‌بازیابی، کل کاندیدا با حالت آغازین و RNG بازیابی‌شده روی CPU تکرار می‌شود. dtype، batch منطقی و تعداد optimizer updateها خودکار عوض نمی‌شوند.

تجمیع sparse جدید و ضرب زوجی با حافظهٔ موقت محدود، همان معادلات و گرادیان مرتبهٔ اول را محاسبه می‌کنند. جابه‌جایی دستگاه یا ترتیب جمع‌های ممیز شناور الزاماً برابری bitwise نمی‌دهد. آزمون CUDA و بنچمارک سرعت باید روی سخت‌افزار واقعی انجام شوند؛ موفقیت CPU جای آن را نمی‌گیرد.

برای بازسازی گزارش بدون آموزش:

~~~powershell
python run_v6.py report --datasets ml-100k --output ./results_v6/develop
~~~
