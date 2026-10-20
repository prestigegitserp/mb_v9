# اجرای V5.2 و بررسی هدف 0.87

این دستورها برای checkout دارای فایل‌های V5.2 هستند. تا پیش از commit طبق دستور کاربر، pull شاخهٔ remote این سورس را دریافت نمی‌کند. در این نوبت محیط Python در دسترس نبود؛ هیچ خروجی آموزش یا تستِ تازه‌ای به این دستورها نسبت داده نشده است.

## ۱. نصب و کنترل نسخه

در VS Code، همان interpreter محیط CUDA خودتان، مثلاً foo311، را انتخاب کنید.

~~~bash
python -m pip install -r requirements_v5_2.txt
python verify_v5_2.py
python -m pytest tests/test_v52_models.py tests/test_v52_goal.py tests/test_v52_workflow.py -q -ra
python -m pytest tests -q -ra
python -c "import sys,torch,v52; print(sys.executable); print(v52.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

ورودی جدید run_v5_2.py است. run_v9.py عمداً تغییر نکرده و V5 را اجرا می‌کند. تست‌های V5.2 نوشته شده‌اند ولی در این تحویل اجرا نشده‌اند؛ ابتدا نتیجهٔ واقعی آن‌ها را روی میزبان ثبت کنید. CI جدید نیز صرفاً آماده است و تا commit اجرا نمی‌شود.

## ۲. طرح و smoke

~~~bash
python paper_v5_2.py --phase plan --dataset ml-100k --output ./paper_v52
python paper_v5_2.py --phase smoke --dataset ml-100k --output ./paper_v52 --device cuda:0
~~~

smoke در پوشهٔ جدا و با بودجهٔ کوچک است و نتیجهٔ مقاله نیست. preset جدید high_capacity_ml100k.json شامل فرضیه‌های آزموده‌نشدهٔ decoder پهن و loss ترکیبی است. جست‌وجو کنترل علمی V5 را هم دارد؛ پارامترهای ضعیف نباید صرفاً به‌خاطر تازه‌بودن برنده فرض شوند.

طرح پیش‌فرض بسیار پرهزینه است:20 outer split، سه model seed و سه K. plan تعداد واقعی را محاسبه می‌کند؛ این تعداد در این نوبت اجرا و اندازه‌گیری نشده است. برای تشخیص اولیه و صرفه‌جویی، تمام مراحل یک مطالعهٔ محدود را با آرگومان‌های ثابت اجرا کنید:

~~~bash
python paper_v5_2.py --phase plan --dataset ml-100k --split-seeds 4101 --model-seeds 11 --k 20 --search-jobs-per-method 4 --max-epochs 80 --baseline-max-epochs 300 --output ./v52_development
python paper_v5_2.py --phase scan --dataset ml-100k --split-seeds 4101 --model-seeds 11 --k 20 --search-jobs-per-method 4 --max-epochs 80 --baseline-max-epochs 300 --output ./v52_development --device cuda:0
~~~

این grid و بودجهٔ محدود، بازتولید کامل یا نتیجهٔ نهایی Q1 نیست. فعلاً برای اصلاح مدل از validation استفاده کنید و مرحلهٔ final این پوشهٔ توسعه را وسیلهٔ تنظیم مجدد مدل نکنید.

## ۳. بازتولید رقبا و جست‌وجوی اصلی

~~~bash
python paper_v5_2.py --phase scan --dataset ml-100k --output ./paper_v52 --device cuda:0
~~~

پیش‌فرض MovieLens، aemc-paper و تقسیم85.5/4.5/10 است. هر مدل همان4500 زوج validation را برای انتخاب دارد. برای روش۷، probability fitting خاموش و NLL/پوشش کالیبره گزارش نمی‌شود. mean این روش و تمام مدل‌های دیگر با همان selection سنجیده می‌شوند.

در AEMC-NE، مقاله تعداد hiddenها، Adam و grid دو lambda را مشخص کرده؛ تطابق با کد رسمی نویسنده و فعال‌ساز/initialization/batch دقیق هنوز تأیید نشده است. کنترل identity-transfer با ReLU دقیقاً h(z)=z را در ابتدا ایجاد می‌کند؛ یک فرضیهٔ مقداردهی اولیه است. item-minibatch یک کنترل optimizer با گرادیان unbiased برای همان objective است؛ به‌عنوان جزئیات قطعی مقاله معرفی نمی‌شود.

تمام64 زوج lambda مستقل برای تنظیم اصلی موجودند. کنترل‌های activation/initialization/batch و preset decoder پهن به grid اضافه شده‌اند. محدودیت search-jobs-per-method عمداً این جست‌وجو را ناقص می‌کند و باید در گزارش نوشته شود. تنظیمات واقعی هر baseline در BASELINE_FIDELITY ثبت می‌شود.

اگر هدف probability calibration روش۷ است، مطالعهٔ جدا با --protocol common انجام دهید؛ در آن حالت نقش‌های selection/fit/hold جدا می‌مانند. نتایج آن پروتکل را با عدد0.8767 مقاله یکسان ننامید. Jester و داده‌های غیر MovieLens به‌صورت پیش‌فرض common هستند.

## ۴. ارزیابی اصلی فقط پس از تثبیت

~~~bash
python paper_v5_2.py --phase final --dataset ml-100k --output ./paper_v52 --device cuda:0
python paper_v5_2.py --phase assess --dataset ml-100k --output ./paper_v52
~~~

تمام config، seedها، K، grid و بودجهٔ scan/final باید یکسان باشند. تغییر تنظیمات baselineها یا عضویت TEST نیز رد می‌شود. validation به TRAIN اضافه نمی‌شود.

assess آموزش یا انتخاب پارامتر انجام نمی‌دهد. برای هر هفت روش:

- از NPZ واقعی TEST، RMSE را دوباره حساب می‌کند.
- زوج‌ها و truth را میان همهٔ مدل‌ها تطبیق می‌دهد.
- شمار واقعی85500/4500/10000، 943کاربر و1682آیتم را کنترل می‌کند.
- config منتخب، source hash و provenance انتخاب فقط با validation را بررسی می‌کند.
- میانگین model seedها را در هر outer split، سپس میانگین20 split را گزارش می‌کند.
- عبور از0.87 و برتری بر هر هفت رقیب از پیش اعلام‌شده را جدا نشان می‌دهد.
- اگر AEMC بیش از0.89 باشد، مرجع ضعیف علامت می‌خورد؛ این آستانه صرفاً tolerance تشخیصی است، نه confidence interval مقاله.
- کنترل decoder-only یا کمتر از20 split، شواهد برتری هندسیِ مقاله اعلام نمی‌شود.

خروجی در target_087/ شامل goal_by_method.csv، paired_deltas.csv، archive_integrity.csv و GOAL_ASSESSMENT.json است. نتیجهٔ موفق عددی به‌تنهایی تازگی یا معنی‌داری آماری را ثابت نمی‌کند. splitهای تصادفی همپوشان مستقل فرض نمی‌شوند.

## ۵. ابلیشن و گزارش

~~~bash
python paper_v5_2.py --phase all --dataset ml-100k --output ./paper_v52 --device cuda:0
python paper_v5_2.py --phase report --dataset ml-100k --output ./paper_v52
python paper_v5_2.py --phase assess --dataset ml-100k --output ./paper_v52
~~~

all کارهای کامل‌شده را resume می‌کند و ابلیشن، K، sparsity و convergence را ادامه می‌دهد. calibration در پروتکل point-only صریحاً کنار گذاشته می‌شود. برای اجرای هر مطالعه به‌صورت جدا، نام phase را عوض و سایر آرگومان‌ها را ثابت نگه دارید.

کنترل‌های تازه شامل حذف decoder پهن، حذف full-history term، حذف residual هندسی، حذف auxiliary مکانیسم و encoder قدیمی با LayerNorm است. کنترل decoder-only صرفاً شاهد یک مدل ساده‌تر است؛ پایین‌بودن خطای آن نباید به نوآوری هندسی نسبت داده شود. ظرفیت، تعداد پارامتر، تعداد update، تعداد target دیده‌شده و زمان واقعی همراه خطا گزارش شوند.

گزارش‌ها مانند V5 شامل Excel، CSV کامل، per-pair، تاریخچه، تنظیمات، دستگاه واقعی و شکل‌های PNG/PDF/SVG هستند. مثال بازترسیم:

~~~bash
python replot_v5_2.py --input ./paper_v52/ml-100k/publication/research_tables.xlsx --sheet metrics_long --where metric=test_RMSE --x method --y value --kind barh --aggregate mean --output ./paper_figures_v52/rmse.png
~~~

این بازترسیم آموزش ندارد. گزینه‌های storage، host-budget-mb، ssd-budget-gb و cache-dir در launcher قابل تنظیم‌اند. تغییر ذخیرهٔ موقت batch منطقی، precision و objective را عوض نمی‌کند؛ تغییر item_batch_size در AEMC یک انتخاب علمی مجزا و ثبت‌شده در config است. CUDA و برابری ذخیرهٔ گراف در V5.2 هنوز روی سخت‌افزار آزموده نشده‌اند.
