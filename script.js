document.addEventListener('DOMContentLoaded', () => {
    const nameInputSection = document.getElementById('name-input-section');
    const orderFormSection = document.getElementById('order-form-section');
    const userNameInput = document.getElementById('user-name-input');
    const saveNameBtn = document.getElementById('save-name-btn');
    const displayName = document.getElementById('display-name');
    const orderForm = document.getElementById('order-form');
    const statusMessage = document.getElementById('status-message');
    const orderItemsContainer = document.getElementById('order-items');
    const addItemBtn = document.getElementById('add-item-btn');
    const clearAllBtn = document.getElementById('clear-all-btn');
    const totalAmountInput = document.getElementById('total-amount');

    const userNameKey = 'peonAppName'; // Key for local storage
    let itemCounter = 0; // Counter for unique item IDs

    // --- Check for saved name ---
    const savedName = localStorage.getItem(userNameKey);

    if (savedName) {
        showOrderForm(savedName);
    } else {
        showNameInput();
    }

    // --- Event Listeners ---
    saveNameBtn.addEventListener('click', () => {
        const name = userNameInput.value.trim();
        if (name) {
            localStorage.setItem(userNameKey, name);
            showOrderForm(name);
        } else {
            alert('Please enter your name.');
        }
    });

    addItemBtn.addEventListener('click', () => {
        addOrderItem();
    });

    clearAllBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all items?')) {
            clearAllItems();
        }
    });

    orderForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const name = localStorage.getItem(userNameKey);
        const orderItems = collectOrderItems();

        if (!name) {
            showStatus('Name is missing. Please refresh and enter your name.', 'error');
            return;
        }

        if (orderItems.length === 0) {
            showStatus('Please add at least one item to your order.', 'error');
            return;
        }

        // Validate all items
        for (let i = 0; i < orderItems.length; i++) {
            const item = orderItems[i];
            if (!item.name.trim() || !item.quantity.trim()) {
                showStatus(`Please fill in all fields for item ${i + 1}.`, 'error');
                return;
            }
        }

        // Get total amount given
        const totalAmount = parseFloat(totalAmountInput.value) || 0;
        if (totalAmount <= 0) {
            showStatus('Please enter the total amount given to Kamran.', 'error');
            return;
        }

        const orderData = {
            name: name,
            items: orderItems,
            total: totalAmount
        };

        try {
            const response = await fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData),
            });

            const result = await response.json();

            if (result.success) {
                showStatus('Order submitted successfully!', 'success');
                clearAllItems();
                addOrderItem(); // Add one empty item for next order
                totalAmountInput.value = ''; // Clear total amount
            } else {
                showStatus(`Submission failed: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('Error submitting order:', error);
            showStatus('An error occurred while submitting.', 'error');
        }
    });

    // --- Dynamic Item Management Functions ---
    function addOrderItem(itemName = '', quantity = '') {
        itemCounter++;
        const itemId = `item-${itemCounter}`;

        const itemRow = document.createElement('div');
        itemRow.className = 'order-item-row';
        itemRow.dataset.itemId = itemId;

        itemRow.innerHTML = `
            <input type="text"
                   placeholder="e.g., Daal Chawal, Coffee, Snacks"
                   value="${itemName}"
                   class="item-name"
                   required>
            <input type="text"
                   placeholder="e.g., 1 plate, 2 cups"
                   value="${quantity}"
                   class="item-quantity"
                   required>
            <button type="button" class="delete-item-btn" onclick="removeOrderItem('${itemId}')">
                ×
            </button>
        `;

        orderItemsContainer.appendChild(itemRow);

        updateDeleteButtons();

        return itemRow;
    }

    function removeOrderItem(itemId) {
        const itemRow = document.querySelector(`[data-item-id="${itemId}"]`);
        if (itemRow) {
            itemRow.remove();
            updateDeleteButtons();
        }
    }

    function clearAllItems() {
        orderItemsContainer.innerHTML = '';
        itemCounter = 0;
        totalAmountInput.value = '';
    }

    function updateDeleteButtons() {
        const deleteButtons = document.querySelectorAll('.delete-item-btn');
        const itemCount = deleteButtons.length;

        deleteButtons.forEach(button => {
            button.disabled = itemCount <= 1;
        });
    }

    function collectOrderItems() {
        const items = [];
        const itemRows = document.querySelectorAll('.order-item-row');

        itemRows.forEach(row => {
            const name = row.querySelector('.item-name').value.trim();
            const quantity = row.querySelector('.item-quantity').value.trim();

            if (name || quantity) {
                items.push({
                    name: name,
                    quantity: quantity
                });
            }
        });

        return items;
    }

    // --- Helper Functions ---
    function showNameInput() {
        nameInputSection.style.display = 'block';
        orderFormSection.style.display = 'none';
        statusMessage.style.display = 'none';
    }

    function showOrderForm(name) {
        displayName.textContent = name;
        nameInputSection.style.display = 'none';
        orderFormSection.style.display = 'block';
        statusMessage.style.display = 'none';

        // Initialize with one empty item
        if (orderItemsContainer.children.length === 0) {
            addOrderItem();
        }
    }

    function showStatus(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = `status ${type}`;
        statusMessage.style.display = 'block';

        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
    }

    // Make removeOrderItem globally accessible
    window.removeOrderItem = removeOrderItem;
});
