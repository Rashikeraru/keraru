document.getElementById("registerForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission
    
    // Get form data
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Store user details in localStorage (you can also use cookies)
    localStorage.setItem("username", username);
    localStorage.setItem("password", password);

    // Redirect to login page
    window.location.href = "login.html";
});
