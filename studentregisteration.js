document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission
    
    // Get form data
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Retrieve user details from localStorage (or cookies)
    var storedUsername = localStorage.getItem("username");
    var storedPassword = localStorage.getItem("password");

    // Check if entered credentials match stored credentials
    if (username === storedUsername && password === storedPassword) {
        // Redirect to dashboard or any other page
        window.location.href = "dashboard.html";
    } else {
        // Show error message or handle invalid credentials
       
