# V7 development — بازسازی از انتشار V6

ورودی این شاخه `run_v7.py` و recipe کامل آن `paper_v7.py` است. این شاخه انتشار V7 نیست. `VERSION`، ورودی پیش‌فرض قبلی، تمام فایل‌های نسخه‌های ۱ تا ۶ و سرشاخه‌های آن‌ها حفظ شده‌اند.

- [راهنمای دقیق نصب، پایلوت سبک، scan، final، ابلیشن، ضرایب loss، زمان و نمودار](docs/V7_EXPERIMENTS_FA.md)
- [بررسی پژوهشی نسخه‌ها، پایلوت ارسالی، معماری‌ها، خطاها و ممیزی رقبا](docs/V7_RESEARCH_REVIEW_FA.md)
- [تنظیمات شش دیتاست](examples/v7)
- [منبع اصلی Method10 با نام عمومی Method8، بدون تغییر](vendor_method10/PROVENANCE.json)
- [شواهد بازخوانی فایل‌ها، تست‌ها و آزمایش‌های توسعه](docs/v7_evidence)
- [نمونهٔ ظاهر جدید شکل با دادهٔ V6 ارسالی؛ نتیجهٔ V7 نیست](docs/v7_evidence/FIGURE_LAYOUT_V6_DATA.svg)
- [جدول بررسی تازهٔ validation با ۲۷ آموزش محدود؛ نتیجهٔ TEST نیست](docs/v7_evidence/ML100K_VALIDATION_PROBES.csv)
- [شکل lossهای V7 با داده و تاریخچهٔ واقعی validation](docs/v7_evidence/V7_VALIDATION_OBJECTIVE_PREVIEW.svg)

در پوشهٔ خالی:

```powershell
git clone --branch v7-development --single-branch https://github.com/prestigegitserp/mb_v9.git .
python -m pip install -r requirements_v7.txt
python verify_v7.py
```

پایلوت ثابت مشابه فرمان قبلی، با پروتکل مشترک:

```powershell
python run_v7.py main --datasets ml-100k --data-root ./datasets --split-seeds 2026 --model-seeds 1 --k 20 --output ./results_v7/pilot_common --v7-device cuda --v7-protocol common --no-epoch-train-rmse --resume
```

پایلوت دارای تنظیم هایپرپارامتر و تولید خودکار گزارش منتخب‌ها:

```powershell
python paper_v7.py --stage develop --profile common --datasets ml-100k --data-root ./datasets --output ./results_v7/tuned_common --device cuda
```

`main` برندهٔ اسکن را بارگذاری نمی‌کند. `develop` اسکن و سپس final را اجرا می‌کند. در مطالعهٔ مرحله‌ای، ابتدا `scan` و سپس `final` با همان پارامترهای مطالعه اجرا می‌شود. lossstudy روی validation است؛ جدول اصلی TEST و تحلیل ابلیشن از همان configهای منتخب قابل ردیابی‌اند. کامل‌ترین راهنمای فرمان‌ها در لینک نخست است.

عدد ۰٫۸۷ روی TEST و شکست‌دادن رقبا هنوز از صحت کد، تست CPU یا نتیجهٔ validation نتیجه نمی‌شود. مطالعهٔ مستقل و چندsplit با پروتکل مشترک، معیار ارزیابی این هدف است.
