# Gemini AI Setup for Urdu PDF Generation

## 🚀 Quick Setup Guide

The Office Ordering System now uses Google's Gemini AI to automatically convert Roman Urdu and English item names to proper Urdu script for Kamran's PDFs.

### Step 1: Get Your Free Gemini API Key

1. **Visit Google AI Studio**: https://makersuite.google.com/app/apikey
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy the generated API key**

### Step 2: Configure the System

1. **Open `config.js` file**
2. **Replace `YOUR_GEMINI_API_KEY`** with your actual API key:

```javascript
const CONFIG = {
    GEMINI_API_KEY: 'AIzaSyC-your-actual-api-key-here', // Replace this
    // ... other settings
};
```

### Step 3: Test the System

1. **Open admin.html** in your browser
2. **Add some test orders** with Roman Urdu items like:
   - "Daal Chawal"
   - "Coffee" 
   - "Lassi"
   - "Biryani"
3. **Click "Download List as PDF"**
4. **Check the generated PDF** - items should appear in Urdu script

## 🎯 How It Works

### AI Translation Examples:
- **"Daal Chawal"** → **"دال چاول"**
- **"Coffee"** → **"کافی"** 
- **"Lassi"** → **"لسی"**
- **"Biryani"** → **"بریانی"**
- **"Tea"** → **"چائے"**

### Smart Features:
- ✅ **Caching**: Avoids repeated API calls for same items
- ✅ **Fallback**: Uses original text if AI fails
- ✅ **Batch Processing**: Converts multiple items in one API call
- ✅ **Script Conversion**: Converts pronunciation, not meaning
- ✅ **User Names**: Also converts user names to Urdu script

## 📄 PDF Output

The generated PDF will have:
- **Header**: "دفتری آرڈرز کی فہرست" (Office Orders List)
- **Columns**: نمبر (Number), نام (Name), کیا لانا ہے (Items), قیمت (Amount)
- **Converted Items**: All items in proper Urdu script
- **Excel-like Table**: Professional formatting for Kamran

## 🔧 Troubleshooting

### If PDF generation fails:
1. **Check API Key**: Make sure it's correctly set in `config.js`
2. **Check Internet**: AI conversion requires internet connection
3. **Check Console**: Open browser developer tools for error messages
4. **Fallback**: System will use original text if AI fails

### If items aren't converting:
1. **Check API Quota**: Free tier has daily limits
2. **Check Item Names**: Very unusual names might not convert well
3. **Manual Override**: System falls back to original text

## 💡 Benefits

### For Kamran:
- ✅ **Easy to Read**: Familiar Urdu script instead of English
- ✅ **Professional**: Clean table format like Excel
- ✅ **Accurate**: AI understands context and pronunciation
- ✅ **Consistent**: Same items always convert the same way

### For Admin:
- ✅ **No Manual Work**: Automatic conversion
- ✅ **Scalable**: Handles any new items automatically
- ✅ **Fast**: Cached results for repeated items
- ✅ **Reliable**: Fallback to original text if needed

## 🆓 Cost

- **Gemini API**: Free tier includes generous daily limits
- **No Additional Cost**: Perfect for office use
- **Efficient**: Caching reduces API calls

## 🎉 Ready to Launch!

Once configured, the system will automatically:
1. **Collect** all item names and user names
2. **Send** to Gemini AI for script conversion
3. **Generate** professional Urdu PDF
4. **Cache** results for future use

Perfect for Kamran who reads Urdu much better than English! 🚀
