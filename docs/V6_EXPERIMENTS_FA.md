# راهنمای مطالعهٔ کامل V6 و اجرای کم‌حجم V5

## ۱. رابطهٔ دقیق مراحل

main یک اجرای تنظیمات پایه است و اسکن جامع انجام نمی‌دهد. scan مجموعه‌ای از مدل‌ها را با تنظیم‌های مختلف روی TRAIN آموزش می‌دهد، epoch را با validation انتخاب می‌کند و سپس برای هر روش، بهترین تنظیم را با validation پیدا می‌کند.

برای روش m و outer split شمارهٔ s:

$$
\widehat c_{m,s}
=\arg\min_{c\in\mathcal C_m}\frac1R\sum_{r=1}^{R}
\operatorname{RMSE}_{\mathrm{val},s}(f_{m,s,r,c}).
$$

اسکن هر split از validation همان split استفاده می‌کند. برنده‌ای که با میانگین validation همهٔ splitها انتخاب شده باشد، ممکن است از طریق همپوشانی تقسیم‌ها اطلاعات test یک split را مصرف کند؛ چنین انتخاب سراسری در این کد انجام نمی‌شود.

final تنظیم منتخب را قفل می‌کند، همان TRAIN را با seedهای ثبت‌شده دوباره آموزش می‌دهد و فقط پس از تثبیت، TEST را ارزیابی می‌کند. validation به TRAIN اضافه نمی‌شود. گزارش اصلی برابر است با میانگین model seedها در هر split، سپس میانگین outer splitها:

$$
\overline E_m=\frac1S\sum_s\frac1R\sum_r
\operatorname{RMSE}_{\mathrm{test},s}(f_{m,s,r,\widehat c_{m,s}}).
$$

~~~mermaid
flowchart TD
    A["TRAIN: آموزش کاندیداها"] --> B["Validation: انتخاب تنظیم و epoch"]
    B --> C["قفل تنظیم هر مدل و هر split"]
    C --> D["Final: ارزیابی test و جدول اصلی"]
    C --> E["ابلیشن و تحلیل حساسیت"]
    D --> F["Excel و شکل‌های نهایی"]
    E --> F
~~~

ابلیشن از تنظیم منتخب همان مدل شروع می‌شود و یک مؤلفه را تغییر می‌دهد. برای هر ابلیشن از نو روی TEST تنظیم انتخاب نمی‌شود. kstudy نیز تنظیم‌های دیگرِ برنده را ثابت می‌گذارد و K را برای تحلیل توصیفی تغییر می‌دهد؛ انتخاب بهترین K باید قبلاً در scan و با validation انجام شده باشد.

در اینجا ثابت‌بودن config به معنی ثابت‌بودن وزن‌های آموزش‌دیده نیست: هر حذف دوباره آموزش می‌بیند، epoch و مرحله‌های داخلی با همان procedure انتخاب می‌شوند. grid بیرونی دوباره جست‌وجو نمی‌شود. ضریب fittedِ response هر seed، خطای خام و خطای اصلاح‌شده در response_corrections.csv و best_configurations.json ثبت می‌شوند.

## ۲. دیتاست‌ها و قالب‌های مستقل

| دیتاست | مسیر ورودی نسبت به data-root | پروتکل پیش‌فرض قالب | رقبا |
|---|---|---|---|
| synthetic | تولید داخلی | common | MF، BiasMF، سه MBMF |
| ml-100k | ml-100k/u.data | aemc-paper | رقبا + AutoRec و AEMC-NE |
| ml-1m | ml-1m/ratings.dat | aemc-paper | رقبا + AutoRec و AEMC-NE |
| ml-10m | ml-10M100K/ratings.dat | aemc-paper | رقبا + AutoRec و AEMC-NE |
| jester | jester/jester-data-1.xls و فایل‌های موجود 2 و 3 | common | MF، BiasMF، سه MBMF |
| bookcrossing | bookcrossing/BX-Book-Ratings.csv | common | MF، BiasMF، سه MBMF |

هر هفت روش پیشنهادی روی شش دیتاست برنامه‌ریزی می‌شوند. ماتریس پشتیبانی فعلی AutoRec/AEMC-NE را به MovieLens محدود می‌کند؛ نبود آن‌ها در Jester یا BookCrossing را با نتیجهٔ ضعیف یا شکست آموزش اشتباه نگیرید.

MovieLens در aemc-paper از تقسیم global random به نسبت 85.5/4.5/10 درصد استفاده می‌کند. در common نسبت اصلی 80/10/10 است؛ به‌صورت پیش‌فرض validation به selection و probability-fit و probability-hold با نسبت 60/20/20 تقسیم می‌شود. همهٔ روش‌ها از population یکسان selection استفاده می‌کنند. در پروتکل paper، Method7 فقط point prediction دارد و برای آن NLL و نمودار پوشش احتمال تولید نمی‌شود.

BookCrossing فقط امتیازهای explicit مثبت را می‌گیرد؛ صفر امتیاز نیست. Jester مقدار 99 را missing می‌داند. RMSE خام را میان مقیاس‌های متفاوت این دیتاست‌ها مستقیماً رتبه‌بندی نکنید؛ معیارهای نرمال‌شده نسبت به دامنه هم در جدول‌های بازحساب‌شده موجودند.

## ۳. اجرای توسعه‌ای

ابتدا فقط حجم برنامه را ببینید:

~~~powershell
python paper_v6.py --stage plan --profile development --datasets all --data-root ./datasets --output ./results_v6/develop
~~~

برای توسعهٔ ML-100K و ساخت خودکار گزارش منتخب‌ها:

~~~powershell
python paper_v6.py --stage develop --profile development --datasets ml-100k --data-root ./datasets --output ./results_v6/develop --device cuda
~~~

این دستور مانند اجرای pilot قبلی یک split و یک model seed و K=20 دارد، ولی چند تنظیم را بررسی می‌کند. فضای development عمداً محدود است و بازتولید جامع مقالهٔ رقیب نیست. برای آزمایش کمتر می‌توان --search-jobs-per-method 4 داد؛ از ابتدا مسیر خروجی متفاوت بگذارید. حذف این محدودیت را با تغییر config در همان مطالعهٔ نیمه‌کاره انجام ندهید.

برای توسعهٔ همهٔ دیتاست‌ها کافی است --datasets all بدهید؛ داده‌های واقعی باید قبلاً آماده باشند. توسعهٔ مدل با دیدن TEST در چند رفت‌وبرگشت، TEST را عملاً به دادهٔ توسعه تبدیل می‌کند؛ برای ادعای نهایی، seedهای از پیش کنارگذاشته‌شده و پروتکل تثبیت‌شده لازم است.

## ۴. مطالعهٔ اصلی مقاله

پروفایل paper از 20 outer split با seedهای 4101 تا 4120، سه model seed برابر 11/22/33 و Kهای 10/20/50 استفاده می‌کند. این بودجه می‌تواند بسیار گران باشد؛ plan را حتماً اول اجرا کنید. نام comprehensive به فضای تعریف‌شدهٔ جست‌وجو اشاره دارد و به معنی تمام ترکیب‌های ممکن هر پارامتر نیست.

~~~powershell
python paper_v6.py --stage plan --profile paper --datasets all --data-root ./datasets --output ./results_v6/paper
python paper_v6.py --stage scan --profile paper --datasets all --data-root ./datasets --output ./results_v6/paper --device cuda
python paper_v6.py --stage final --profile paper --datasets all --data-root ./datasets --output ./results_v6/paper --device cuda
~~~

scan در launcher با scan-only اجرا می‌شود؛ در این مرحله test ارزیابی نمی‌شود. final به search/validation_winners.csv و configهای آرشیوشدهٔ همان مسیر وابسته است. dataset، grid، seed، مدل‌ها، config، بودجه و سورس علمی باید در این دو مرحله یکسان بمانند. انتخاب CPU/CUDA و تنظیم‌های اجرا هم در config و provenance ثبت می‌شوند.

فرمان مستقیم run_v6.py scan به‌صورت پیش‌فرض final را هم انجام می‌دهد؛ برای توقف قبل از TEST باید --scan-only بدهید. این تفاوت با launcher عمدی و در نام stageها مشخص است.

## ۵. آزمایش‌های جامع و ارتباطشان

همهٔ فرمان‌های زیر همان آرگومان‌های profile، dataset، output و هر override مراحل scan/final را باید حفظ کنند:

| آزمایش | stage | نقطهٔ شروع و نقش |
|---|---|---|
| جست‌وجوی validation و حساسیت تنظیم‌ها | scan | تولید کاندیدا، رتبه‌بندی، برندهٔ مستقل هر split |
| جدول اصلی دقت و کارایی | final | فقط تنظیم منتخب scan؛ test برای گزارش |
| حذف مؤلفه‌های هر روش | ablation | برندهٔ همان مدل، با K منتخب همان split |
| حذف مؤلفه‌ها در K یکسان | ablation-fixed | تنظیم منتخب، K=20 برای full و همهٔ حذف‌ها |
| حساسیت به K | kstudy | تنظیم منتخب، تغییر K؛ نتایج test صرفاً توصیفی |
| کاهش TRAIN با held-out ثابت | sparsity | تنظیم منتخب؛ validation/test ثابت می‌مانند |
| تغییر چگالی دادهٔ مصنوعی | density | فقط synthetic؛ مطالعهٔ داده، نه hyperparameter انتخابی |
| منحنی‌های همگرایی | convergence | فقط train/validation با schedule برنده، بدون test هر epoch |
| معیار جایگزین احتمالاتی Method7 | calibration | مطالعهٔ جدا با نقش‌های disjoint در common |
| تقسیم زمانی | temporal | فقط MovieLens؛ مطالعه‌ای با تقسیم و selection مستقل |
| بازسازی شکل و Excel | report | بدون آموزش |
| ممیزی انتخاب‌ها | audit | بررسی provenance و جدایی validation از test |

برای ابلیشن، K، sparsity، calibration و convergence فایل publication/research_tables.xlsx مستقل هم ساخته می‌شود. variantها به‌عنوان seedهای تکراری یک روش با هم میانگین نمی‌شوند. density از configهای پایهٔ از پیش مشخص‌شده استفاده می‌کند و K/init را با validation همان داده انتخاب می‌کند؛ این مطالعهٔ تولید داده را با مقایسهٔ fixed-test یا تنظیم دوبارهٔ نتیجهٔ اصلی یکسان نگیرید. temporal فضای جست‌وجوی مستقل خودش را دارد و مرجع AEMC با split تصادفی به آن نسبت داده نمی‌شود.

~~~powershell
python paper_v6.py --stage ablation --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage ablation-fixed --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage kstudy --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage sparsity --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage convergence --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage calibration --profile paper --datasets all --output ./results_v6/paper --device cuda
python paper_v6.py --stage density --profile paper --datasets synthetic --output ./results_v6/paper --device cuda
python paper_v6.py --stage temporal --profile paper --datasets ml-100k,ml-1m,ml-10m --output ./results_v6/paper --device cuda
python paper_v6.py --stage report --profile paper --datasets all --output ./results_v6/paper
python paper_v6.py --stage audit --profile paper --datasets all --output ./results_v6/paper
~~~

در مسیر data-root غیرپیش‌فرض، همان --data-root را در همهٔ فرمان‌ها اضافه کنید. اگر --methods یا override بودجه داده‌اید، آن را هم در همهٔ مراحل حفظ کنید.

برای اجرای متوالی همهٔ مراحل اصلی:

~~~powershell
python paper_v6.py --stage all --profile paper --datasets all --data-root ./datasets --output ./results_v6/paper --device cuda
~~~

all شامل density برای synthetic است. temporal فقط با --include-temporal اضافه می‌شود. calibration در MovieLens paper به‌دلیل point-only بودن Method7 صریحاً کنار گذاشته می‌شود. آزمایش‌های احتمالاتی MovieLens باید مطالعهٔ common جدا با output تازه باشند و عدد آن‌ها به‌عنوان بازتولید Table 2 معرفی نشود.

## ۶. چه چیزی را در مقاله گزارش کنم؟

جدول اصلی را از final_locked_test_summary.csv و publication/tables/metric_summary.csv بردارید؛ میانگین model seedها ابتدا در هر split گرفته می‌شود. تمام روش‌های از پیش اعلام‌شده را گزارش کنید، نه فقط روشی که روی TEST بهتر دیده شده است. یک split و یک seed برای برتری آماری کافی نیست.

V6 برخلاف summary عمومی زیرساخت قدیمی، برنده‌های K یا initialization متفاوت را به سطرهای جدا نمی‌شکند. هر روش در جدول اصلی یک سطر دارد؛ فهرست Kهای منتخب و تعداد outer split/model seed کنار آن ثبت می‌شود. std روی میانگین‌های outer split حساب می‌شود، نه با فرض مستقل‌بودن همهٔ model seedها.

تنظیم‌ها را از search/best_configurations.json، search/validation_winners.csv و publication/tables/selected_configurations.csv بردارید. source_lineage نشان می‌دهد مدل منتخب از کدام معماری آمده است. اگر تنظیم در splitها فرق دارد، آن را مخفی نکنید؛ جدول فراوانی انتخاب یا فایل supplement کامل بدهید.

حساسیت‌ها از search/publication/tables/hyperparameter_sensitivity.csv و search/figures هستند. نقطه‌های این نمودارها configهای کامل‌اند؛ این‌ها لزوماً اثر علّیِ جداگانهٔ یک پارامتر را نشان نمی‌دهند.

ابلیشن، ظرفیت، تعداد پارامتر، optimizer step، targetهای دیده‌شده، زمان و دستگاه واقعی را کنار دقت گزارش کنید. انتخاب بهترین نتیجه از نمودار test حساسیت، انتخاب مجدد روی test است و در این مسیر مجاز نیست.

splitهای تصادفیِ تکرارشده همپوشان‌اند؛ جدول‌های Wilcoxon/Friedman خروجی، exploratory هستند و استقلال نمونه‌ها یا تأیید نهایی آماری را ثابت نمی‌کنند. پیروزی بر یک baseline ضعیف، بدون ممیزی fidelity، شاهد کافی نیست.

## ۷. نمودار از Excel

~~~powershell
python replot_v6.py --input ./results_v6/paper/ml-100k/publication/research_tables.xlsx --sheet metrics_long --where metric=test_RMSE --x method --y value --kind barh --aggregate mean --output ./paper_figures/ml100k_rmse.png
~~~

این دستور آموزش ندارد. CSVها، تاریخچهٔ epoch و per-pairها برای شکل‌های دیگر حفظ می‌شوند. در خروجی تک‌split، خطای معیار تکرارهای مستقل ساخته نمی‌شود. در پروتکل بدون probability، نمودار احتمال جعلی تولید نمی‌شود.

## ۸. دستور بهینه‌تر برای V5 منتشرشدهٔ فعلی

این بخش به هیچ فایل جدید V6 وابسته نیست و در checkout فعلی V5 اجرا می‌شود:

~~~powershell
python run_v5_2.py scan --datasets ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v5/tuned_pilot_4 --v52-device cuda --v52-config ./examples/v52/high_capacity_ml100k.json --scan-mode one-factor --v52-search-jobs-per-method 4 --v52-retain-candidates metrics --resume
~~~

در کد V5 فعلی، plan این مطالعه را با 51 job و 76 آموزش کاندیدا گزارش می‌کند؛ محدودیت چهار job برای هر روش به معنی چهار آموزش کل نیست. initializationهای MBMF و ادغام بعضی jobها باعث تفاوت می‌شوند. این یک جست‌وجوی محدود توسعه‌ای است، نه grid کامل مقالهٔ AEMC-NE.

این فرمان، برنده‌ها را با validation انتخاب می‌کند و سپس خودکار final، شکل‌ها و Excel را تولید می‌کند. --scan-only ندهید اگر همین رفتار را می‌خواهید. در صورت قطع، همان فرمان با resume تکرار شود. برای گزارش تنها:

~~~powershell
python run_v5_2.py report --datasets ml-100k --output ./results_v5/tuned_pilot_4
~~~

retain-candidates=metrics در V5 فقط checkpoint کاندیداها را حذف می‌کند؛ پیش‌بینی‌ها و گزارش‌های تمام trialها هنوز ذخیره می‌شوند. پس کاهش حجم دارد، ولی سیاست نگهداری رهبران V6 را ندارد. با محدودکردن تعداد job و K، حجم V5 کنترل می‌شود. اسکن هم باگ‌های علمی تشخیص‌داده‌شدهٔ V5 را اصلاح نمی‌کند.
