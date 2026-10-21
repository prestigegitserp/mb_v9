# رفع خطای CSV بزرگ هنگام ساخت Excel نسخه۵٫۲

اگر بعد از `20/20 FINISHED` و پایان stage این خطا دیده می‌شود:

```text
v52/export.py ... for row in reader
_csv.Error: field larger than field limit (131072)
```

آموزش تمام شده و ساخت workbook متوقف شده است. فیلدهای JSON مربوط به config یا تشخیص فازها می‌توانند از محدودیت پیش‌فرض parser ماژول csv بزرگ‌تر باشند. منطق ارجاع سلول‌های بزرگ Excel از قبل وجود داشت، اما parser پیش از رسیدن به آن متوقف می‌شد.

اصلاح فقط در `v52/export.py` است: سقف parser هنگام خواندن هر CSV موقتاً تا حد سازگار با C long در Python3.11 روی Windows بالا می‌رود؛ مقدار بزرگ‌تر موجود کاهش پیدا نمی‌کند و تنظیم قبلی در پایان یا خطا برمی‌گردد. مقدار کامل در CSV حفظ می‌شود؛ سلول‌های بزرگ‌تر از۳۲۷۶۷ کاراکتر در Excel همچنان به محل مقدار کامل ارجاع می‌دهند. کد مدل‌ها، آموزش، checkpointها، split و predictionها تغییری نمی‌کند. منبع نسخه‌های۱ تا۴ و namespace قبلی `v5/` تغییر نمی‌کند.

## بازیابی گزارش بدون تکرار آموزش

پس از دریافت اصلاح، از checkout مخزن اجرا کنید. مقدار output باید ریشه‌ای باشد که `ml-100k/main` داخل آن است:

```powershell
python run_v5_2.py report --datasets ml-100k --output ./results_v5/pilot_2
```

این مسیر CSVها و predictionهای موجود را می‌خواند و فایل‌های قابل بازسازی گزارش، شکل‌ها و Excel را دوباره می‌سازد. به اجرای `main` یا اضافه‌کردن `--resume` برای بازسازی گزارش نیاز نیست. مخصوصاً job ثبت‌شده با وضعیت FAILED ممکن است با `main --resume` وارد اجرای مجدد job شود. تغییر سورس نیز می‌تواند fingerprint را عوض کند؛ فرمان report برای همین بازیابی از خروجی ذخیره‌شده است. پوشهٔ نتایج یا CSVها را حذف نکنید.

اگر output متفاوت بوده، همان مسیر واقعی را جایگزین کنید. فایل نهایی Excel معمولاً در این مسیر است:

```text
results_v5/pilot_2/ml-100k/main/publication/research_tables.xlsx
```

## راه‌حل فوری روی کد قبلی

این دستور، فقط سقف csv را در همان process بالا می‌برد و command گزارش را اجرا می‌کند؛ فایلی از سورس را تغییر نمی‌دهد:

```powershell
python -c "import csv,runpy,sys; csv.field_size_limit(2147483647); sys.argv=['run_v5_2.py','report','--datasets','ml-100k','--output','./results_v5/pilot_2']; runpy.run_path('run_v5_2.py',run_name='__main__')"
```

پیام `DataFrame is highly fragmented` هشدار عملکرد pandas و پیام font weight هشدار جایگزینی فونت است؛ traceback فعلی از آن‌ها نیست. ساخت مجدد گزارش آموزش را تکرار نمی‌کند، اما ترسیم و export بزرگ می‌تواند زمان ببرد.

## آزمون هدفمند

```powershell
python -m unittest discover -s tests -p test_v52_export_csv.py -v
```

تست‌ها فیلد بزرگ‌تر از۱۳۱۰۷۲، مرز۳۲۷۶۷ Excel، متن فارسی quoted و چندخطی، حفظ CSV و precision شناسه، سازگاری limit با Windows و بازگردانی limit در خطا را بررسی می‌کنند. آزمون‌های فوق به ساخت مدل یا CUDA نیاز ندارند.
