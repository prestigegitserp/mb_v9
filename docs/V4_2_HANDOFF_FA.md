# تحویل نسخهٔ ۴٫۲ برای شاخهٔ V4

این بسته تغییرات **uncommitted** شاخهٔ موجود `v4-development` را تحویل می‌دهد. مبنا `3b1af9306a9ef9da541c5b0fabbe74dbafeb3011` است. کد به GitHub push نشده و commit یا merge انجام نشده است. شش شاخهٔ remote و ۶۶ فایل محافظت‌شدهٔ نسخه‌های قبلی بدون تغییر باقی مانده‌اند.

## اعمال روی کپی خودتان

ZIP را خارج از پوشهٔ repository باز کنید. با Python این فرمان را اجرا کنید و مسیر repository خودتان را بدهید:

```bash
python /path/to/extracted_bundle/APPLY_V4_2.py --repo /path/to/mb_v9
```

در ویندوز مسیرهای دارای فاصله را داخل کوتیشن بگذارید:

```powershell
python "D:\work\v4_2_delivery\APPLY_V4_2.py" --repo "D:\work\mb_v9"
```

اسکریپت پیش از نوشتن فایل‌ها، hash سورس بسته، پایهٔ Git، نبود تغییر staged، فایل‌های محافظت‌شده و conflict با ویرایش‌های محلی را کنترل می‌کند. روی `v4-development` موجود کار می‌کند؛ روی شاخهٔ دیگر، فقط در working tree تمیز به آن شاخه می‌رود. اگر آن شاخه محلی موجود نباشد، از remote موجود یا همین پایهٔ مشخص، فقط همان نام شاخه را می‌سازد. شاخهٔ پژوهشی اضافه و تغییر سرِ remote ندارد. اگر خودتان V4 را روی پایهٔ متفاوت commit کرده‌اید، آن را خودکار بازنویسی نمی‌کند.

دو نقطهٔ شروع پشتیبانی می‌شوند: پایهٔ تمیزِ مشخص‌شده، یا overlay دست‌نخوردهٔ تحویل پیشین ۴٫۱/۴٫۰. ویرایش محلی متعارض باعث توقف پیش از نوشتن سورس می‌شود؛ آن را با force دور نزنید. apply هیچ فایل نامرتبطی را پاک نمی‌کند و مرحله‌ای برای stage، commit، merge یا push ندارد.

`uncommitted_v4_2.patch` برای پایهٔ **تمیز** است. برای repository دارای overlay قبلی، اسکریپت apply را به کار ببرید؛ اجرای دوبارهٔ patch کامل روی فایل‌های از قبل اضافه‌شده تعارض طبیعی می‌سازد. اعمال خودکار روی پایهٔ تمیز و تحویل ۴٫۱ در دو کپی جدا آزمایش شده و `PATCH_REPLAY.json` نتیجه را ثبت می‌کند.

پس از اعمال، در repository اجرا کنید:

```bash
python verify_v4.py --require-uncommitted-base
python -m pip install -r requirements_v4.txt
python paper_v4_2.py --phase plan --dataset ml-100k --output ./paper_v4_2
```

برای PyTorch مناسبِ کارت CUDA، نصب سازگار با دستگاه خودتان لازم است. آزمون ذخیره/بازیابیِ GPU را پیش از مطالعهٔ سنگین اجرا کنید:

```bash
python experiments/v4_execution_parity.py --device cuda:0 --steps 3 --threads 2 --output ./gpu_parity_check --cache-dir ./v4_ssd_cache
```

اجرای پژوهش، مرحله به مرحله، در `source_v4/docs/V4_2_PAPER_WORKFLOW_FA.md` توضیح داده شده است. `paper_v4_2.py --phase scan` عمداً پیش از TEST می‌ایستد؛ `final` برنده‌های همان output را ارزیابی می‌کند؛ `all` مراحل تکمیلی را با resume ادامه می‌دهد. `smoke` جدا و غیرقابل استفاده برای مقاله است.

## بازخوانی وزن‌های آماده

`checkpoints/` شامل وزن‌های منتخب هفت مدل و هفت رقیب است، همراه آرشیو پیش‌بینی و provenance. این‌ها متعلق به split=2026 و seed=1 هستند؛ وزن‌ها جای اجرای کامل چند split/seed مقاله را نمی‌گیرند. هیچ TEST واقعی تازه‌ای در این تحویل ارزیابی نشده است.

برای بارگذاری یک مدل V4، داده را دقیقاً از `data/split_assignments_2026.csv` بسازید و نقش‌های validation را با `v4.protocol.validation_roles` و split=2026 اعمال کنید؛ سپس `v4.checkpoints.load` را استفاده کنید. امضای داده و سورس بررسی می‌شود؛ صرف هم‌اندازه‌بودن ماتریس کافی نیست. نمونهٔ کامل ساخت داده و بارگذاری در `source_v4/experiments/v4_2_verify_release_weights.py` آمده است. baselineهای native با loader هستهٔ ثابت و MBMF با `v4.baselines.restore_mbmf` بازخوانی می‌شوند.

حاشیهٔ مهم: وزن‌های قبلی با دادهٔ ولیدیشنِ ثابت و اختلاف پیش‌بینی صفر به فرمت انتشار تطبیق یافته‌اند؛ بازآموزی تازه نامیده نشده‌اند و hash سورس زمان آموزش در provenance محفوظ است. `evidence/frozen_sources/` سورس‌های واقعیِ زمان آن آزمایش‌ها را نگه می‌دارد. مسیرهای `WINNER_SOURCES.json` نسبت به ریشهٔ همین بسته‌اند.

برای یافته‌های واقعی و رقبا، `source_v4/docs/V4_2_REVIEW_FA.md`؛ برای نظریه و ایده‌های اجرا‌نشده، `source_v4/docs/V4_2_THEORY_FA.md` را ببینید. جدول‌ها و شکل‌های این تحویل از **ولیدیشن اکتشافی** هستند و نتیجهٔ نهایی TEST مقاله نیستند.
