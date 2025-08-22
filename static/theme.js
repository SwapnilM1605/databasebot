document.addEventListener('DOMContentLoaded', () => {
    const toggleButtons = document.querySelectorAll('.theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'day';
    
    // Apply the saved theme on page load
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateToggleIcons(currentTheme);

    // Toggle theme on button click
    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const newTheme = document.documentElement.getAttribute('data-theme') === 'day' ? 'night' : 'day';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateToggleIcons(newTheme);
        });
    });

    // Update icon based on theme
    function updateToggleIcons(theme) {
        toggleButtons.forEach(button => {
            button.innerHTML = theme === 'day' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
        });
    }
});