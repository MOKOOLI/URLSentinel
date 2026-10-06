<div dir="rtl">

# راهنمای فارسی URLSentinel

## این پروژه چیه؟

یه سیستم **تشخیص لینک فیشینگ** با شبکه عصبی که **از صفر با NumPy** نوشته شده و برای هر تصمیم توضیح می‌ده چرا (Shapley values).
**هیچ داده مصنوعی‌ای نداره:**

- آموزش روی دیتاست واقعی **PhiUSIIL** (حدود ۲۳۵ هزار URL واقعی، مقاله Computers & Security 2024)
- تست روی **فیشینگ‌های زنده امروز** از فید OpenPhish، یعنی لینک‌هایی که چند ساعت پیش گزارش شدن

## اجرا توی Git Bash (ویندوز)

پیش‌نیاز: Python 3.9+ (موقع نصب تیک **Add Python to PATH**)، Git for Windows، و اینترنت (بار اول حدود ۵۵ مگ دانلود می‌کنه).

```bash
cd ~/Desktop/urlsentinel
./run.sh            # نصب، دانلود دیتا، آموزش، تست زنده، دمو
./run.sh serve      # رابط وب: http://127.0.0.1:8000
./run.sh live       # دوباره تست روی فیشینگ‌های امروز
./run.sh test       # تست‌ها
```

آموزش کامل روی لپ‌تاپ معمولی چند دقیقه طول می‌کشه. برای سریع‌تر: `./run.sh train --limit 50000`

بعد از اجرا، **README خودش با نتایج واقعی تو پر می‌شه** (جدول دقت، نرخ تشخیص زنده، خروجی دمو). همونو commit کن.

## گذاشتن روی گیت‌هاب

۱. توی github.com یه ریپوی خالی به اسم `urlsentinel` بساز (بدون README).

۲. توی README، `YOUR_USERNAME` رو با یوزرنیم خودت عوض کن.

۳. توی Git Bash:

```bash
git init
git add .
git commit -m "feat: URLSentinel, explainable phishing detection with a NumPy neural network"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/urlsentinel.git
git push -u origin main
```

۴. تب **Actions**: تست‌ها روی لینوکس و ویندوز اجرا می‌شن، و هر دوشنبه مدل خودکار دوباره آموزش می‌بینه و روی فیشینگ همون روز تست می‌شه.

## برای مصاحبه

این سه تا نکته تو رو از بقیه جدا می‌کنه، حتماً بلد باش توضیحشون بدی:

- **Shortcut audit**: توی PhiUSIIL بیشتر لینک‌های سالم فقط صفحه اصلی سایتن، پس مدل می‌تونه تقلب کنه و یاد بگیره «لینک بدون مسیر = سالم». پروژه اینو خودش پیدا و گزارش می‌کنه.
- **Split بر اساس hostname**: هیچ سایتی هم توی train هم توی test نیست، پس عدد دقت باد نکرده.
- **تست out-of-time**: دیتای آموزش مال ۲۰۲۲-۲۳ـه، تست زنده مال امروز. این عدد واقعی‌ترین عدده.

topicهای ریپو: `machine-learning` `cybersecurity` `phishing-detection` `numpy` `explainable-ai` `python`

</div>
