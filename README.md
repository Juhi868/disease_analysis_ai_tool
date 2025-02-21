# disease_analysis_ai_tool
basic AI-powered rare disease diagnostic tool
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Rare Diagnostic Tool</title>
    <style>
        /* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Poppins', sans-serif;
}

body {
    background-color: #f5f5f5;
}

/* Navigation Bar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 10%;
    background: #1c1c1e;
    color: white;
    position: fixed;
    width: 100%;
    top: 0;
    z-index: 1000;
}

.logo {
    font-size: 24px;
    font-weight: 600;
}

.nav-links {
    list-style: none;
    display: flex;
}

.nav-links li {
    margin: 0 15px;
}

.nav-links a {
    text-decoration: none;
    color: white;
    font-weight: 400;
    transition: 0.3s;
}

.nav-links a:hover {
    color: #4ade80;
}

.cta-btn {
    background: #4ade80;
    padding: 10px 15px;
    border-radius: 5px;
    font-weight: 600;
}

.cta-btn:hover {
    background: #3cc36b;
}

/* Hero Section */
.hero {
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: linear-gradient(135deg, #0d7377, #14ffec);
    color: white;
    padding: 20px;
    margin-top: 70px; /* To prevent overlap with fixed navbar */
}

.hero-content h1 {
    font-size: 48px;
    font-weight: 600;
}

.hero-content p {
    font-size: 18px;
    margin: 15px 0;
    opacity: 0.9;
}

.cta-buttons {
    margin-top: 20px;
}

.primary-btn, .secondary-btn {
    text-decoration: none;
    padding: 12px 24px;
    border-radius: 5px;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s ease-in-out;
    display: inline-block;
}

.primary-btn {
    background: #4ade80;
    color: #1c1c1e;
    margin-right: 15px;
}

.primary-btn:hover {
    background: #3cc36b;
}

.secondary-btn {
    background: transparent;
    color: white;
    border: 2px solid white;
}

.secondary-btn:hover {
    background: white;
    color: #1c1c1e;
}

    </style>
</head>
<body>

    <!-- Navigation Bar -->
    <header>
        <nav class="navbar">
            <div class="logo">AI Diagnosis</div>
            <ul class="nav-links">
                <li><a href="#">Home</a></li>
                <li><a href="#">How It Works</a></li>
                <li><a href="#">Features</a></li>
                <li><a href="#">Contact</a></li>
                <li><a href="#" class="cta-btn">Get Started</a></li>
            </ul>
        </nav>
    </header>

    <!-- Hero Section -->
    <section class="hero">
        <div class="hero-content">
            <h1>Revolutionizing Rare Disease Diagnosis with AI</h1>
            <p>Fast, accurate, and early detection powered by cutting-edge artificial intelligence.</p>
            <div class="cta-buttons">
                <a href="#" class="primary-btn">Try It Now</a>
                <a href="#" class="secondary-btn">Request a Demo</a>
            </div>
        </div>
    </section>

</body>
</html>

