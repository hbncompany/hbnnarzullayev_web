$(document).ready(function() {
    // Show loading animation when a button or link is clicked (before MySQL execution)
    $('#your-button-or-link').click(function() {
        $('#loading').show();
    });

    // Hide loading animation when MySQL execution is complete (for example, when data is loaded)
    // You need to adapt this part according to your actual implementation
    // This is just a placeholder example
    $.ajax({
        url: '/your-mysql-execution-route',
        method: 'GET',
        success: function(response) {
            // Hide loading animation
            $('#loading').hide();
            // Do something with the response (update the page content)
        }
    });
});

document.addEventListener("DOMContentLoaded", function() {
    var openPopupButton = document.getElementById("open-popup");
    var closePopupButton = document.getElementById("close-popup");
    var popup = document.getElementById("popup");

    openPopupButton.addEventListener("click", function() {
        popup.style.display = "flex";
    });

    closePopupButton.addEventListener("click", function() {
        popup.style.display = "none";
    });
});
const togglePassword = document.querySelector('#togglePassword');
  const password = document.querySelector('#id_password');

  togglePassword.addEventListener('click', function (e) {
    // toggle the type attribute
    const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
    password.setAttribute('type', type);
    // toggle the eye slash icon
    this.classList.toggle('fa-eye-slash');
});