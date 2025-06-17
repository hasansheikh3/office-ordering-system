// Helper function to properly handle timestamp conversion
function parseTimestamp(timestampStr) {
    // If the timestamp doesn't include timezone info, treat it as local time
    if (timestampStr && !timestampStr.includes('T') && !timestampStr.includes('Z')) {
        // SQLite format: "YYYY-MM-DD HH:MM:SS"
        // Convert to ISO format and treat as local time
        const isoString = timestampStr.replace(' ', 'T');
        return new Date(isoString);
    }
    return new Date(timestampStr);
}

document.addEventListener('DOMContentLoaded', async () => {
    const adminNameSection = document.getElementById('admin-name-section');
    const adminNameInput = document.getElementById('admin-name-input');
    const adminNameSubmitBtn = document.getElementById('admin-name-submit');
    const adminNameStatus = document.getElementById('admin-name-status');

    const adminContentSection = document.getElementById('admin-content-section');
    const adminList = document.getElementById('admin-list');
    const downloadBtn = document.getElementById('download-btn');
    const deleteAllBtn = document.getElementById('delete-all-btn'); // New
    const newOrderBtn = document.getElementById('new-order-btn');     // New
    const loadingStatus = document.getElementById('loading-status');
    const errorStatus = document.getElementById('error-status');
    const captureArea = document.getElementById('capture-area');
    const captureTotal = document.getElementById('capture-total');

    const adminNewOrderSection = document.getElementById('admin-new-order-section'); // New
    const adminOrderForm = document.getElementById('admin-order-form');         // New
    const adminNameField = document.getElementById('admin-name');       // New
    const adminItemField = document.getElementById('admin-item');       // New
    const adminQuantityField = document.getElementById('admin-quantity'); // New
    const adminAmountField = document.getElementById('admin-amount');     // New
    const cancelNewOrderBtn = document.getElementById('cancel-new-order'); // New
    const adminOrderStatus = document.getElementById('admin-order-status'); // New


    const adminName = 'Kamran'; // Hardcoded admin name

    // Initially show the name input section and hide content
    adminNameSection.style.display = 'block';
    adminContentSection.style.display = 'none';
    adminNewOrderSection.style.display = 'none'; // Hide new order form too

    // --- Event Listener for Admin Name Submission ---
    adminNameSubmitBtn.addEventListener('click', () => {
        const enteredName = adminNameInput.value.trim();

        if (enteredName === adminName) {
            // Correct name entered
            adminNameSection.style.display = 'none';
            adminContentSection.style.display = 'block';
            fetchOrders(); // Fetch orders now that access is granted
        } else {
            // Incorrect name
            showAdminNameStatus('Incorrect name.', 'error');
        }
    });

    // Allow pressing Enter in the admin name input field
    adminNameInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            adminNameSubmitBtn.click();
        }
    });

    // --- Event Listener for New Order Button ---
    newOrderBtn.addEventListener('click', () => {
        adminContentSection.style.display = 'none';
        adminNewOrderSection.style.display = 'block';
        adminOrderForm.reset(); // Clear the form
        adminOrderStatus.style.display = 'none'; // Hide status
        adminNameField.focus(); // Put focus on the first field
    });

    // --- Event Listener for Cancel New Order Button ---
    cancelNewOrderBtn.addEventListener('click', () => {
        adminNewOrderSection.style.display = 'none';
        adminContentSection.style.display = 'block';
        fetchOrders(); // Refresh list when returning
    });

    // --- Event Listener for Admin Order Form Submission ---
    adminOrderForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const name = adminNameField.value.trim();
        const item = adminItemField.value.trim();
        const quantity = adminQuantityField.value.trim();
        const amount = adminAmountField.value.trim();

         if (!name || !item || !quantity || !amount) {
            showAdminOrderStatus('Please fill in all fields.', 'error');
            return;
        }

        const orderData = {
            name: name,
            item: item,
            quantity: quantity,
            amount: amount
        };

        try {
            // Send POST request to the same /submit endpoint
            // Add admin name for backend check (simple validation)
            const response = await fetch(`/submit?name=${adminName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData),
            });

            const result = await response.json();

            if (result.success) {
                showAdminOrderStatus('Order submitted successfully!', 'success');
                // After successful submission, hide the form and show the list
                // Use a slight delay to allow user to see the success message
                setTimeout(() => {
                    adminNewOrderSection.style.display = 'none';
                    adminContentSection.style.display = 'block';
                    fetchOrders(); // Refresh the list
                }, 1000); // 1 second delay
            } else {
                showAdminOrderStatus(`Submission failed: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('Error submitting admin order:', error);
            showAdminOrderStatus('An error occurred while submitting.', 'error');
        }
    });


    // --- Fetch Orders Function ---
    async function fetchOrders() {
        loadingStatus.style.display = 'block';
        errorStatus.style.display = 'none';
        adminList.innerHTML = ''; // Clear previous list

        try {
            // Include the admin name in the request for backend validation
            const response = await fetch(`/admin/data?name=${adminName}`);
            const result = await response.json();

            loadingStatus.style.display = 'none';

            if (result.success) {
                displayOrders(result.orders);
            } else {
                errorStatus.textContent = `Error: ${result.message}`;
                errorStatus.style.display = 'block';
            }
        } catch (error) {
            console.error('Error fetching orders:', error);
            loadingStatus.style.display = 'none';
            errorStatus.textContent = 'An error occurred while fetching orders.';
            errorStatus.style.display = 'block';
        }
    }

    // --- Display Orders Function ---
    function displayOrders(orders) {
        let totalAmount = 0;
        adminList.innerHTML = ''; // Clear list

        if (orders.length === 0) {
            adminList.innerHTML = '<li class="no-orders">No orders submitted yet.</li>';
            captureTotal.textContent = 'Total Amount: 0';
             // Hide elements if no orders
            captureArea.querySelector('.capture-footer').style.display = 'none';
            return;
        }

        captureArea.querySelector('.capture-footer').style.display = 'block'; // Show footer if orders exist

        // Group orders by user and timestamp (to group multi-item orders)
        const groupedOrders = groupOrdersByUser(orders);

        groupedOrders.forEach(userGroup => {
            const orderCard = document.createElement('li');
            orderCard.className = 'order-card';

            // Create items list for this user's order
            let itemsHtml = '';
            let orderIds = [];
            let totalUserAmount = 0;
            let timestamps = [];

            // Track unique session amounts to avoid double counting
            const sessionAmounts = new Set();

            userGroup.orders.forEach((order) => {
                orderIds.push(order.id);
                timestamps.push(parseTimestamp(order.timestamp));

                itemsHtml += `
                    <div class="order-item">
                        <span class="item-name">${order.item}</span>
                        <span class="item-quantity">Qty: ${order.quantity}</span>
                        <span class="item-time">${parseTimestamp(order.timestamp).toLocaleString()}</span>
                    </div>
                `;

                // Calculate total amount for this user - sum unique session totals
                const amountValue = parseFloat(order.amount.replace(/[^0-9.-]+/g,""));
                if (!isNaN(amountValue)) {
                    // Use timestamp to group orders from the same session (within 1 minute)
                    const orderTime = parseTimestamp(order.timestamp).getTime();
                    const sessionKey = Math.floor(orderTime / 60000); // Group by minute

                    // Only add amount if we haven't seen this session before
                    if (!sessionAmounts.has(sessionKey)) {
                        sessionAmounts.add(sessionKey);
                        totalUserAmount += amountValue;
                    }
                }
            });

            // Create time range display
            const earliestTime = new Date(Math.min(...timestamps));
            const latestTime = new Date(Math.max(...timestamps));
            let timeDisplay = '';

            if (timestamps.length === 1) {
                timeDisplay = latestTime.toLocaleString();
            } else {
                timeDisplay = `${earliestTime.toLocaleString()} - ${latestTime.toLocaleString()}`;
            }

            orderCard.innerHTML = `
                <div class="order-header">
                    <div class="order-user">
                        <strong>${userGroup.name}</strong>
                        <span class="order-time">${timeDisplay}</span>
                        <span class="order-count">${userGroup.orders.length} item(s)</span>
                    </div>
                    <div class="order-actions">
                        <button class="delete-order-btn" data-ids="${orderIds.join(',')}" title="Delete all orders for ${userGroup.name}">
                            <span class="delete-icon">×</span>
                        </button>
                    </div>
                </div>
                <div class="order-items">
                    ${itemsHtml}
                </div>
                <div class="order-footer">
                    <span class="order-amount">Total: Rs ${totalUserAmount.toFixed(2)}</span>
                </div>
            `;

            adminList.appendChild(orderCard);

            // Add to grand total
            totalAmount += totalUserAmount;
        });

        // Display total amount
        captureTotal.textContent = `Total Amount: ${totalAmount.toFixed(2)}`;
    }

    // --- Group Orders by User Function ---
    function groupOrdersByUser(orders) {
        const groups = {};

        orders.forEach(order => {
            // Group by name only (all orders from same user together)
            const groupKey = order.name;

            if (!groups[groupKey]) {
                groups[groupKey] = {
                    name: order.name,
                    orders: []
                };
            }

            groups[groupKey].orders.push(order);
        });

        // Sort orders within each group by timestamp (most recent first)
        Object.values(groups).forEach(group => {
            group.orders.sort((a, b) => {
                const timeA = parseTimestamp(a.timestamp).getTime();
                const timeB = parseTimestamp(b.timestamp).getTime();
                return timeB - timeA;
            });
        });

        // Convert to array and sort groups by most recent order in each group
        return Object.values(groups).sort((a, b) => {
            const timeA = parseTimestamp(a.orders[0].timestamp).getTime();
            const timeB = parseTimestamp(b.orders[0].timestamp).getTime();
            return timeB - timeA;
        });
    }

    // --- Event Delegation for Delete Buttons ---
    adminList.addEventListener('click', (event) => {
        // Check if the clicked element is a delete button or its child
        const deleteBtn = event.target.closest('.delete-order-btn');
        if (deleteBtn) {
            const orderIds = deleteBtn.dataset.ids.split(',');
            const userName = deleteBtn.closest('.order-card').querySelector('.order-user strong').textContent;

            if (confirm(`Are you sure you want to delete ${userName}'s order?`)) {
                deleteOrders(orderIds);
            }
        }
    });

    // --- Function to Delete Multiple Orders ---
    async function deleteOrders(orderIds) {
        try {
            // Delete each order individually
            const deletePromises = orderIds.map(orderId =>
                fetch(`/delete/${orderId}?name=${adminName}`, {
                    method: 'DELETE',
                })
            );

            const responses = await Promise.all(deletePromises);
            const results = await Promise.all(responses.map(r => r.json()));

            // Check if all deletions were successful
            const allSuccessful = results.every(result => result.success);

            if (allSuccessful) {
                console.log('All orders deleted successfully');
                fetchOrders(); // Refresh the list after deletion
            } else {
                console.error('Some deletions failed:', results);
                alert('Some orders could not be deleted. Please try again.');
            }
        } catch (error) {
            console.error('Error deleting orders:', error);
            alert('An error occurred while deleting the orders.');
        }
    }

    // --- Event Listener for Delete All Button ---
    deleteAllBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to delete ALL orders? This cannot be undone.')) {
            deleteAllOrders();
        }
    });

     // --- Function to Delete All Orders ---
    async function deleteAllOrders() {
        try {
            // Add admin name for backend check (simple validation)
            const response = await fetch(`/delete_all?name=${adminName}`, {
                method: 'DELETE',
            });

            const result = await response.json();

            if (result.success) {
                console.log(result.message);
                fetchOrders(); // Refresh the list after deletion
            } else {
                console.error(`Delete All failed: ${result.message}`);
                alert(`Failed to delete all orders: ${result.message}`);
            }
        } catch (error) {
            console.error('Error deleting all orders:', error);
            alert('An error occurred while deleting all orders.');
        }
    }


    // --- Download Button Handler - Generate PDF ---
    downloadBtn.addEventListener('click', () => {
        generatePDF();
    });

    // --- Gemini AI Agent Configuration ---
    // Note: Replace 'YOUR_GEMINI_API_KEY' in config.js with your actual Gemini API key
    // Get your free API key from: https://makersuite.google.com/app/apikey
    const GEMINI_API_KEY = CONFIG?.GEMINI_API_KEY || 'YOUR_GEMINI_API_KEY';
    const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

    // Cache for translated items to avoid repeated API calls
    const translationCache = new Map();

    // --- Fallback Translation Dictionary (for when AI is not available) ---
    const fallbackTranslations = {
        // Food items
        'daal chawal': 'دال چاول',
        'daal': 'دال',
        'chawal': 'چاول',
        'coffee': 'کافی',
        'tea': 'چائے',
        'chai': 'چائے',
        'pani': 'پانی',
        'water': 'پانی',
        'roti': 'روٹی',
        'naan': 'نان',
        'biryani': 'بریانی',
        'karahi': 'کڑاہی',
        'kebab': 'کباب',
        'samosa': 'سموسہ',
        'lassi': 'لسی',
        'paratha': 'پراٹھا',
        'cigarette': 'سگریٹ',
        'ciggarate': 'سگریٹ',

        // Names
        'hasan': 'حسن',
        'hassan': 'حسن',
        'junaid': 'جنید',
        'ibrahim': 'ابراہیم',
        'ahmad': 'احمد',
        'ali': 'علی',
        'kamran': 'کامران'
    };

    // --- Fallback function for when AI is not available ---
    function fallbackTranslateToUrdu(text) {
        if (!text) return text;

        const lowerText = text.toLowerCase().trim();
        return fallbackTranslations[lowerText] || text;
    }

    // --- AI Agent: Convert Roman Urdu/English to Urdu Script ---
    async function convertToUrduScript(items, userNames) {
        // Check if API key is configured
        if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY') {
            console.log('Gemini API key not configured, using fallback translations');
            return {
                items: items.map(item => fallbackTranslateToUrdu(item)),
                names: userNames.map(name => fallbackTranslateToUrdu(name))
            };
        }

        try {
            // Prepare unique items and names for translation
            const uniqueItems = [...new Set(items)];
            const uniqueNames = [...new Set(userNames)];
            const allTexts = [...uniqueItems, ...uniqueNames];

            // Filter out already cached items
            const uncachedTexts = allTexts.filter(text => !translationCache.has(text.toLowerCase()));

            if (uncachedTexts.length === 0) {
                // All items are cached, return cached results
                return {
                    items: items.map(item => translationCache.get(item.toLowerCase()) || fallbackTranslateToUrdu(item)),
                    names: userNames.map(name => translationCache.get(name.toLowerCase()) || fallbackTranslateToUrdu(name))
                };
            }

            // Create prompt for Gemini AI
            const prompt = `You are an expert in converting Roman Urdu and English words to Urdu script. Your task is to convert the given words to Urdu script WITHOUT translating their meaning.

IMPORTANT RULES:
1. Convert the SOUND/PRONUNCIATION to Urdu script, NOT the meaning
2. For example: "Coffee" should become "کافی" (kaafi), not "قہوہ" (qahwa)
3. For example: "Lassi" should become "لسی", not any other Urdu word for yogurt drink
4. For example: "Daal Chawal" should become "دال چاول"
5. Keep the same pronunciation but write it in Urdu script
6. If it's already a proper English word that doesn't have a Roman Urdu equivalent, keep it as is

Convert these words to Urdu script (one per line, in the same order):
${uncachedTexts.join('\n')}

Respond with only the Urdu script versions, one per line, in the exact same order. No explanations or additional text.`;

            const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{
                            text: prompt
                        }]
                    }],
                    generationConfig: {
                        temperature: 0.1,
                        maxOutputTokens: 1000,
                    }
                })
            });

            if (!response.ok) {
                console.log(`Gemini API error: ${response.status}`);
                console.log(response);
                throw new Error(`Gemini API error: ${response.status}`);
            }

            const data = await response.json();
            const aiResponse = data.candidates[0].content.parts[0].text.trim();
            console.log(aiResponse);
            const convertedTexts = aiResponse.split('\n').map(line => line.trim());

            // Cache the results
            uncachedTexts.forEach((originalText, index) => {
                if (convertedTexts[index]) {
                    translationCache.set(originalText.toLowerCase(), convertedTexts[index]);
                }
            });

            // Return converted items and names
            return {
                items: items.map(item => translationCache.get(item.toLowerCase()) || fallbackTranslateToUrdu(item)),
                names: userNames.map(name => translationCache.get(name.toLowerCase()) || fallbackTranslateToUrdu(name))
            };

        } catch (error) {
            console.error('Error converting to Urdu script, using fallback:', error);
            // Fallback: use fallback translations
            return {
                items: items.map(item => fallbackTranslateToUrdu(item)),
                names: userNames.map(name => fallbackTranslateToUrdu(name))
            };
        }
    }

    // --- Font Loading Function for Urdu Support ---
    async function loadUrduFont(doc) {
        // For now, we'll use a simpler approach with HTML5 Canvas to render Urdu text as images
        // This ensures proper Urdu rendering without needing to embed large font files
        return Promise.resolve(); // Placeholder for now
    }

    // --- Helper function to check if text contains Urdu characters ---
    function isUrduText(text) {
        if (!text) return false;
        // Check for Arabic/Urdu Unicode range (U+0600 to U+06FF)
        const urduRegex = /[\u0600-\u06FF]/;
        return urduRegex.test(text);
    }

    // --- Helper function to render Urdu text to PDF ---
    async function renderUrduTextToPDF(doc, text, x, y, fontSize = 10) {
        try {
            // Create a temporary div to render the Urdu text with proper fonts
            const tempDiv = document.createElement('div');
            tempDiv.style.position = 'absolute';
            tempDiv.style.left = '-9999px';
            tempDiv.style.top = '-9999px';
            tempDiv.style.fontSize = `${fontSize}px`;
            tempDiv.style.fontFamily = '"Noto Sans Arabic", Arial, sans-serif';
            tempDiv.style.color = 'black';
            tempDiv.style.backgroundColor = 'white';
            tempDiv.style.padding = '2px 4px';
            tempDiv.style.whiteSpace = 'nowrap';
            tempDiv.textContent = text;

            document.body.appendChild(tempDiv);

            // Use html2canvas to capture the text as image
            if (window.html2canvas) {
                const canvas = await html2canvas(tempDiv, {
                    backgroundColor: 'white',
                    scale: 2
                });

                // Convert canvas to image data
                const imgData = canvas.toDataURL('image/png');

                // Calculate dimensions for PDF
                const imgWidth = canvas.width / 4; // Scale down for PDF
                const imgHeight = canvas.height / 4;

                // Add image to PDF
                doc.addImage(imgData, 'PNG', x, y - fontSize/2, imgWidth, imgHeight);
            } else {
                // Fallback: use regular text if html2canvas is not available
                doc.text(text, x, y);
            }

            // Clean up
            document.body.removeChild(tempDiv);

        } catch (error) {
            console.warn('Error rendering Urdu text, using fallback:', error);
            // Fallback to regular text
            doc.text(text, x, y);
        }
    }

    // --- Generate PDF Function with AI-Powered Urdu Support ---
    async function generatePDF() {
        const { jsPDF } = window.jspdf;

        // Show loading indicator
        downloadBtn.textContent = 'Generating PDF...';
        downloadBtn.disabled = true;

        try {
            // Collect all items and user names for AI conversion
            const orders = Array.from(adminList.querySelectorAll('.order-card'));
            const allItems = [];
            const allUserNames = [];

            orders.forEach(orderCard => {
                const userName = orderCard.querySelector('.order-user strong').textContent;
                allUserNames.push(userName);

                const items = orderCard.querySelectorAll('.order-item');
                items.forEach(item => {
                    const itemName = item.querySelector('.item-name').textContent;
                    allItems.push(itemName);
                });
            });

            // Use AI agent to convert to Urdu script
            const converted = await convertToUrduScript(allItems, allUserNames);

            // Create PDF with Unicode support
            const doc = new jsPDF();

            // Add Noto Sans Arabic font for Urdu support
            // This is a base64 encoded version of Noto Sans Arabic Regular
            // You can generate this using: https://peckconsulting.s3.amazonaws.com/fontconverter/fontconverter.html
            try {
                // Try to load the Urdu font from external file
                await loadUrduFont(doc);
            } catch (error) {
                console.warn('Could not load Urdu font, using fallback approach:', error);
            }
            const now = new Date();

            // PDF Configuration
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 15;
            let yPosition = margin;

            // Header in English (to avoid font issues)
            doc.setFontSize(20);
            doc.setFont(undefined, 'bold');
            doc.text('Office Orders List (Kamran)', pageWidth / 2, yPosition, { align: 'center' });
            yPosition += 8;

            doc.setFontSize(12);
            doc.setFont(undefined, 'normal');
            doc.text(`Date: ${now.toLocaleDateString()} Time: ${now.toLocaleTimeString()}`, pageWidth / 2, yPosition, { align: 'center' });
            yPosition += 15;

            // Table headers
            const colWidths = {
                sno: 15,      // S.No
                naam: 45,     // Name
                items: 80,    // Items
                qeemat: 35    // Amount
            };

            const colPositions = {
                sno: margin,
                naam: margin + colWidths.sno,
                items: margin + colWidths.sno + colWidths.naam,
                qeemat: margin + colWidths.sno + colWidths.naam + colWidths.items
            };

            // Draw table header
            doc.setFillColor(240, 240, 240);
            doc.rect(margin, yPosition, pageWidth - 2 * margin, 12, 'F');

            doc.setFontSize(12);
            doc.setFont(undefined, 'bold');
            doc.text('No.', colPositions.sno + 5, yPosition + 8);

            // Use Urdu headers for better readability by Kamran
            await renderUrduTextToPDF(doc, 'نام', colPositions.naam + 5, yPosition + 8, 12);
            await renderUrduTextToPDF(doc, 'کیا لانا ہے', colPositions.items + 5, yPosition + 8, 12);
            await renderUrduTextToPDF(doc, 'قیمت', colPositions.qeemat + 5, yPosition + 8, 12);

            // Draw header border
            doc.setDrawColor(0, 0, 0);
            doc.rect(margin, yPosition, pageWidth - 2 * margin, 12);
            yPosition += 12;

            // Get grouped orders data
            const orderCards = Array.from(adminList.querySelectorAll('.order-card'));
            let rowNumber = 1;
            let grandTotal = 0;

            if (orderCards.length === 0) {
                doc.setFontSize(14);
                doc.text('No orders found', pageWidth / 2, yPosition + 20, { align: 'center' });
            } else {
                doc.setFont(undefined, 'normal');
                doc.setFontSize(10);

                // Process each order group with AI-converted text
                for (let cardIndex = 0; cardIndex < orderCards.length; cardIndex++) {
                    const orderCard = orderCards[cardIndex];
                    // Check if we need a new page
                    if (yPosition > pageHeight - 50) {
                        doc.addPage();
                        yPosition = margin;

                        // Redraw header on new page
                        doc.setFillColor(240, 240, 240);
                        doc.rect(margin, yPosition, pageWidth - 2 * margin, 12, 'F');
                        doc.setFontSize(12);
                        doc.setFont(undefined, 'bold');
                        doc.text('No.', colPositions.sno + 5, yPosition + 8);

                        // Use Urdu headers for better readability by Kamran
                        await renderUrduTextToPDF(doc, 'نام', colPositions.naam + 5, yPosition + 8, 12);
                        await renderUrduTextToPDF(doc, 'کیا لانا ہے', colPositions.items + 5, yPosition + 8, 12);
                        await renderUrduTextToPDF(doc, 'قیمت', colPositions.qeemat + 5, yPosition + 8, 12);

                        doc.rect(margin, yPosition, pageWidth - 2 * margin, 12);
                        yPosition += 12;
                        doc.setFont(undefined, 'normal');
                        doc.setFontSize(10);
                    }

                    // Get order data
                    const userName = orderCard.querySelector('.order-user strong').textContent;
                    const orderAmount = orderCard.querySelector('.order-amount').textContent;
                    const items = orderCard.querySelectorAll('.order-item');

                    // Get converted user name
                    const convertedUserName = converted.names[cardIndex] || userName;

                    // Extract amount value
                    const amountMatch = orderAmount.match(/[\d.]+/);
                    const amountValue = amountMatch ? parseFloat(amountMatch[0]) : 0;
                    grandTotal += amountValue;

                    // Prepare items text with AI conversion
                    let itemsText = '';
                    items.forEach((item, itemIndex) => {
                        const itemName = item.querySelector('.item-name').textContent;
                        const itemQuantity = item.querySelector('.item-quantity').textContent.replace('Qty: ', '');

                        // Find the converted item name
                        const itemIndexInAll = allItems.indexOf(itemName);
                        const convertedItem = converted.items[itemIndexInAll] || itemName;

                        if (itemIndex > 0) itemsText += '\n';

                        // Handle RTL text direction for Urdu - put quantity before the Urdu text
                        if (isUrduText(convertedItem)) {
                            itemsText += `(${itemQuantity}) ${convertedItem}`;
                        } else {
                            itemsText += `${convertedItem} (${itemQuantity})`;
                        }
                    });

                    // Calculate row height based on items - give more space for multiple items
                    const itemLines = itemsText.split('\n');
                    const rowHeight = Math.max(15, itemLines.length * 12 + 8); // Increased spacing for better readability

                    // Draw row background (alternating colors)
                    if (rowNumber % 2 === 0) {
                        doc.setFillColor(250, 250, 250);
                        doc.rect(margin, yPosition, pageWidth - 2 * margin, rowHeight, 'F');
                    }

                    // Draw row content
                    doc.text(rowNumber.toString(), colPositions.sno + 5, yPosition + 8);

                    // For Urdu text, use a different approach to ensure proper rendering
                    if (isUrduText(convertedUserName)) {
                        // Use HTML element to render Urdu text properly, then capture as image
                        await renderUrduTextToPDF(doc, convertedUserName, colPositions.naam + 5, yPosition + 8, 10);
                    } else {
                        doc.text(convertedUserName, colPositions.naam + 5, yPosition + 8);
                    }

                    // Draw items (multi-line support)
                    if (isUrduText(itemsText)) {
                        // Handle multi-line Urdu text
                        const urduLines = itemsText.split('\n');
                        for (let lineIndex = 0; lineIndex < urduLines.length; lineIndex++) {
                            const lineY = yPosition + 8 + (lineIndex * 12);
                            await renderUrduTextToPDF(doc, urduLines[lineIndex], colPositions.items + 5, lineY, 10);
                        }
                    } else {
                        // Handle multi-line English text
                        const itemsLines = itemsText.split('\n');
                        for (let lineIndex = 0; lineIndex < itemsLines.length; lineIndex++) {
                            const lineY = yPosition + 8 + (lineIndex * 12);
                            doc.text(itemsLines[lineIndex], colPositions.items + 5, lineY);
                        }
                    }

                    doc.text(`Rs ${amountValue.toFixed(0)}`, colPositions.qeemat + 5, yPosition + 8);

                    // Draw row border
                    doc.setDrawColor(200, 200, 200);
                    doc.rect(margin, yPosition, pageWidth - 2 * margin, rowHeight);

                    yPosition += rowHeight;
                    rowNumber++;
                }

                // Draw total row
                yPosition += 5;
                doc.setFillColor(220, 220, 220);
                doc.rect(margin, yPosition, pageWidth - 2 * margin, 12, 'F');

                doc.setFontSize(12);
                doc.setFont(undefined, 'bold');
                doc.text('Total Amount:', colPositions.items + 5, yPosition + 8);
                doc.text(`Rs ${grandTotal.toFixed(0)}`, colPositions.qeemat + 5, yPosition + 8);

                doc.setDrawColor(0, 0, 0);
                doc.rect(margin, yPosition, pageWidth - 2 * margin, 12);
            }

            // Footer
            const footerY = pageHeight - 15;
            doc.setFontSize(8);
            doc.setFont(undefined, 'normal');
            doc.setTextColor(128, 128, 128);
            doc.text('Generated by Office Ordering System', pageWidth / 2, footerY, { align: 'center' });

            // Save the PDF
            doc.save('kamran_orders.pdf');

        } catch (error) {
            console.error('Error generating PDF:', error);
            alert('Error generating PDF. Please try again.');
        } finally {
            // Reset button
            downloadBtn.textContent = 'Download List as PDF';
            downloadBtn.disabled = false;
        }
    }

    // --- Helper function for admin name input status ---
    function showAdminNameStatus(message, type) {
        adminNameStatus.textContent = message;
        adminNameStatus.className = `status ${type}`; // 'status success' or 'status error'
        adminNameStatus.style.display = 'block';
    }

    // --- Helper function for admin order status ---
     function showAdminOrderStatus(message, type) {
        adminOrderStatus.textContent = message;
        adminOrderStatus.className = `status ${type}`; // 'status success' or 'status error'
        adminOrderStatus.style.display = 'block';
    }

    // No initial fetchOrders call here, it happens after successful name entry
});
