# نسخه۸.۰.۰؛ انتشار v8 و توسعه v8_development

- [راهنمای انتشار: معادل کامل اجراهای راهنمای نسخه۷، در سه سطح](docs/V8_RELEASE_RUNBOOK_FA.md)
- [جدول تطبیق نسخه۷→۸، معنی همهٔ stageها و مرجع گزینه‌ها](docs/V8_COMMAND_REFERENCE_FA.md)

این شاخه از انتشار `v7@cbc8d2fb36c7abf599ccb8d743497f3322ec4e7a` شروع شده است. تمام ۵۴۸ فایل موجود در درخت آن انتشار، از جمله فایل‌ها و ورودی‌های نسخه‌های۱تا۷، بدون تغییر باقی می‌مانند. ورودی جدید `run_v8.py` و راه‌انداز پیشنهادی `paper_v8.py` است. هشت نام عمومی `Method1` تا `Method8` و هفت baseline حفظ شده‌اند؛ Method8 همان منبع اصلی Method10 بدون تغییر است.

- [راهنمای قدم‌به‌قدم از smoke تا TEST و مجموعهٔ مقاله](docs/V8_RUNBOOK_FA.md)
- [معماری، ضرایب لاس، سهم هندسه و محدودیت ادعاها](docs/V8_MODEL_REVIEW_FA.md)
- [آزمون‌ها و نتایج واقعی توسعه](docs/V8_VALIDATION_FA.md)
- [نوت‌بوک آمادهٔ مراحل](notebooks/V8_Research_Workflow_FA.ipynb)
- [قالب‌های سه پروفایل و شش دیتاست](examples/v8)

دریافت در پوشهٔ جدا از نسخه۷:

```bash
git clone --branch v8 --single-branch https://github.com/prestigegitserp/mb_v9.git mb_v9_v8
cd mb_v9_v8
python -m pip install -r requirements_v8.txt
python verify_v8.py
```

## روشن‌ترین مسیر تنظیم و TEST

```bash
python paper_v8.py --stage scan --profile common --budget light --datasets ml-100k --data-root ./datasets --output ./results_v8/tuned_light --device cuda
python paper_v8.py --stage test --output ./results_v8/tuned_light
python paper_v8.py --stage status --output ./results_v8/tuned_light
```

در فرمان دوم، تنظیمات مطالعهٔ اول خودکار از `V8_RECIPE.json` خوانده می‌شود. `test` نام روشن‌تر `final` است. `scan` فقط TRAIN/VAL است؛ `test` پیش‌بینی TEST، جدول اصلی و نمودارهای TEST می‌سازد. هر دو مرحله با `--stage develop` در یک فرمان هم اجرا می‌شوند.

پس از موفقیت TEST:

- جدول: `results_v8/tuned_light/ml-100k/final_locked_test_summary.csv`
- شکل اصلی: `results_v8/tuned_light/ml-100k/publication/figures/selected_test_rmse.png`
- سایر شکل‌های TEST، در PNG/PDF/SVG: همان پوشهٔ `publication/figures/`
- Excel: `results_v8/tuned_light/ml-100k/publication/research_tables.xlsx`
- وضعیت واقعی تکمیل: `results_v8/tuned_light/ml-100k/MANUSCRIPT_CHECKLIST.md`

اگر `--no-plots` استفاده کرده‌اید:

```bash
python paper_v8.py --stage report --output ./results_v8/tuned_light --plots
```

این فرمان آموزش نمی‌دهد. اگر TEST اجرا نشده باشد، نمودار TEST از دادهٔ VAL ساخته نمی‌شود.

نسخه۸ خطای initialization در finalِ MBMF را با یکسان‌سازی **هویت مؤثر هر کاندیدا** رفع می‌کند؛ کنترل تطابق optimizer، پارامترها و داده خاموش نشده است. مشکل fragmentation گزارش و وزن فونت نیز رفع شده‌اند. بهبود مدل‌های۱تا۷ شامل اعتماد آموختنی به correction هندسی و تنظیم مشترک پاسخ/هندسه روی VAL است. برازش احتمال متد۷ پس از منجمدشدن میانگین انجام می‌شود. ضرایب لاس در registry و گزارش پوشش جست‌وجو مشخص‌اند و ضریب جهت فعال متد۷ نیز ثبت می‌شود.

بهبود VAL، smoke موفق یا تکمیل فایل‌ها، تضمین برتری TEST یا پذیرش Q1 نیست. نتایج توسعه با همان برچسب منتشر می‌شوند؛ ادعای مقاله نیاز به baseline قوی، تکرار و تحلیل سهم سازوکار دارد.

## انتشار، ادامهٔ توسعه و فرمان‌های معادل نسخه۷

شاخهٔ `v8` برای دریافت نسخهٔ انتشار و `v8_development` برای ادامهٔ توسعه است. نصب در پوشهٔ خالی، به‌روزرسانی با pull، آرایه‌های PowerShell به نام V8Light/V8Paper، اجرای همهٔ شش دیتاست، manual seed، منابع و تخمین زمان در راهنمای انتشار آمده‌اند. هر دو فرمان `paper_v8.py --stage curves` و `run_v8.py curves` برای رسم history موجود در دسترس‌اند و آموزش تازه نمی‌دهند.

اصلاح انتشار، نمایش فرمان‌های Windows را با quoting همان محیط سازگار می‌کند، از تحمیل smoke به recipe پژوهشی قبلی جلوگیری می‌کند و مسیر فایل مرجع `estimate_v8.py` را به شواهد موجودِ نسخه۷ وصل می‌کند. برای اجراهای شروع‌شده با سورس توسعهٔ قبلی، خروجی تازه انتخاب کنید.
