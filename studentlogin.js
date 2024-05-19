// Function to read and parse CSV data
function readCSV(file) {
    // Read file contents asynchronously
    fetch(file)
        .then(response => response.text())
        .then(data => {
            // Parse CSV data
            let rows = data.split('\n').slice(1); // Exclude header row
            let users = [];
            rows.forEach(row => {
                let [username, password] = row.split(',');
                users.push({ username, password });
            });
            // Call function to authenticate user
            authenticateUser(users);
        })
        .catch(error => console.error('Error reading file:', error));
}

// Function to authenticate user
function authenticateUser(users) {
    // Get input values from login form
    let usernameInput = document.getElementById('username').value;
    let passwordInput = document.getElementById('password').value;

    // Check if provided credentials match any user in the CSV data
    let isAuthenticated = users.some(user => user.username === usernameInput && user.password === passwordInput);

    // Handle authentication result
    if (isAuthenticated) {
        alert('Login successful!');
        // Redirect or perform desired action upon successful login
    } else {
        alert('Invalid username or password. Please try again.');
    }
}

// Function to handle form submission
function handleSubmit(event) {
    event.preventDefault(); // Prevent form from submitting
    readCSV('1.csv'); // Read and parse CSV file
}

// Add event listener to form submit event
document.getElementById('loginForm').addEventListener('submit', handleSubmit);
