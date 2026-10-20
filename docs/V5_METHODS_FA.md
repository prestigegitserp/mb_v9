# مکانیسم، کنترل و ادعای قابل ابطال V5

این سند با گردش‌کار مهارت `deep-theoretical-fusion-engine` تهیه شده است: مکانیسم، شرط کاربرد، کنترل خلاف‌واقع و نتیجهٔ منفی صریح‌اند. انتقال ابزار از یک حوزه به توصیه‌گر به‌تنهایی اثبات نوآوری نیست. هیچ نتیجهٔ کلی دربارهٔ برتری یا پذیرش Q1 از نام‌گذاری روش‌ها حاصل نمی‌شود. مرجع اجرایی `v5/models.py` و تنظیمات کامل هر fit است.

## قرارداد مشترک

برای query هدف، context فقط یال‌های باقی‌ماندهٔ TRAIN است؛ rating و timestamp خود query به encoder وارد نمی‌شود. statistics، شمار مشاهده، posterior و graph probe از همین context دوباره محاسبه می‌شوند. دامنهٔ پیش‌بینی از ساختار و bound کنترل می‌شود. uncertainty روش‌های۳ و۵ تقریب شرطی با تعداد probe محدود است، نه posterior دقیق تمام پارامترهای شبکه.

encoder جدید یک مسیر صفرآغاز برای میانگین residual و log-count دارد. در حالت degree-gated، اعتماد با ضریب `sqrt(n/(n+tau))` تعدیل می‌شود. وزن‌های این مسیر با RNG محلی ساخته می‌شوند تا initialization قدیمی ناخواسته جابه‌جا نشود. در پیش‌فرض روش۱ و۵ degree gate خاموش است؛ حذف residual skip و روشن/خاموش‌کردن degree gate کنترل‌های مستقل‌اند.

| روش | تغییر اجراشده و منشأ مکانیسم | کنترل خلاف‌واقع و محدودیت |
|---|---|---|
| ۱ | encoder residual در cavity اطلاعاتی؛ normalization وابسته به degree به‌عنوان فرضیهٔ conditioning | normalization در validation بدتر شد و پیش‌فرض نیست؛ spherical geometry و کنترل gradient interference باقی‌اند؛ descent واقعی Adam تضمین نمی‌شود |
| ۲ | gate آموخته از OOF با shrinkage پشتیبانی به gate ثابت؛ اعتبارسنجی strength دقیقاً همان gate تعدیل‌شدهٔ inference را می‌سنجد | `no_support_shrinkage`، gate ثابت، expert/decoder null؛ OOF مدل‌ها مستقل از نظر آماری فرض نمی‌شوند |
| ۳ | اصلاح شعاع با Gauss–Newton محدود و line search شامل گام صفر؛ نمونه‌گیری graph شرطی | `no_radius_line_search`، global radius و no graph uncertainty؛ فقط surrogate context در گام پذیرفته بدتر نمی‌شود، تضمین خطای query/VAL نیست |
| ۴ | prior نامتقارن برای partial moment مثبت/منفی، آمیخته با آمار context به نسبت `n/(n+tau)`؛ projection مخروط مجاز و readout خطی | `symmetric_amplitude_prior`، حذف refinement، کنترل متریک اقلیدسی/وزنی و decoder-only؛ اثر مشترک L2 و readout باید با factorial جدا شود |
| ۵ | readout coupled با mean decoder و پاسخ Gaussian مشترک ماتریس‌آزاد؛ prior precision وابسته به degree در ridge، Hessian و noise یکسان پیاده شده است | کنترل degree-scaled از validation سود نگرفت و پیش‌فرض خاموش است؛ plugin mean، posterior مستقل و covariance صفر نیز موجودند؛ SPD و PCG tolerance کنترل می‌شوند |
| ۶ | انتقال angular همراه shrinkage به نگاشت تحلیلی در گروه مرجع کم‌نمونه: `rho=N/(N+tau)` | `no_transport_support_shrinkage`، analytic/no transport و referenceهای جایگزین؛ N تعداد نمونهٔ reference است، گواه confidence آماری نیست |
| ۷ | bootstrap نمایی temperشده `w=(-log U)^kappa` با kappa=0.5 و loss میانگین ensemble؛ مرحلهٔ توزیع پس از تثبیت میانگین | kappa=1، عضو منفرد، بدون mean loss و shape محدود؛ mixture variance و برابری میانگین احتمال با prediction آزمون دارند |

## حسابرسی ادعا و ابطال

**آزموده در این میزبان:** عدم ورود query به context؛ آموزش با TEST مسموم؛ bound خروجی؛ checkpoint/prediction یکسان؛ ثابت‌بودن میانگین probability؛ برابری update و AdamW هنگام ذخیرهٔ گراف روی CPU/RAM/SSD؛ تولید کامل فایل‌های scan/final/report؛ رد config دستکاری‌شده پس از انتخاب. جزئیات شمار آزمون در `V5_VERIFICATION.json` است.

**شاهد تجربی محدود:** نتایج ۳۵ آموزش validation با یک تقسیم و یک seed. جدول review تمام تفاوت‌های بودجه و نتایج منفی را اعلام می‌کند. AutoRec قوی‌تر در جدول رقبا باقی مانده است. نسبت‌دادن بهبود کل به یک مکانیسم، بدون ابلیشن، مجاز نیست.

**فرضیهٔ نیازمند آزمون مستقل:** degree shrinkage بیشترین سود را برای context کم‌نمونه می‌دهد؛ line search نرخ گام نامفید شعاع را کم می‌کند؛ prior نامتقارن tail error را بهتر از prior متقارن کاهش می‌دهد؛ tempering ensemble افت variance را با افت calibration عوض نمی‌کند. ابطال با برابری/بدترشدن نسبت به کنترل در splitهای مستقل، matched-K و گزارش فاصلهٔ خطا/بودجه انجام می‌شود. نتیجهٔ منفی باید باقی بماند.

سه مسیر پژوهشی با سطح ادعای متفاوت داریم: مسیر عملی شامل همین اصلاح conditioning، گزارش و کنترل اجرای آزموده است؛ مسیر اصلی مقاله ترکیب partial moments، metric projection و readout با ابلیشن factorial است؛ مسیر بلندمدت، contraction و پوشش calibrated برای عملگر context-dependent است. دو مسیر آخر هنوز اثبات یا تأیید مستقل ندارند و قابلیت پیاده‌سازی فعلی به معنی حل آن‌ها نیست.

## تعهدات نظری باز

۱. bound نقطه‌ای با calibration پوشش احتمالی یکی نیست. ۲. خطای PCG و finite probes باید در عدم‌قطعیت گزارش شوند. ۳. پذیرفتن line search فقط objective تعریف‌شدهٔ context را کنترل می‌کند. ۴. gradient surgery روی مختصات encoder ضمانت descent با momentum و weight decay نیست. ۵. shrinkage پشتیبانی با افزایش داده باید به مسیر یادگرفته‌شده میل کند، ولی بهینه‌بودن tau اثبات نشده است. ۶. برای ادعای تازگی جهانی، مرور prior art فراتر از baselineهای فعلی لازم است؛ جست‌وجوی محدود این نوبت چنین ادعایی را توجیه نمی‌کند.

پیش از نگارش نتیجهٔ قوی، readout و regularization را در یک طراحی مشترک، با decoder-only و baseline هم‌ظرفیت مقایسه کنید؛ سپس هزینهٔ واقعی، حساسیت به degree، tail error و failureها را گزارش دهید. اگر کنترل ساده همان بهبود را ایجاد کرد، ادعا باید به همان مکانیسم ساده محدود شود.
