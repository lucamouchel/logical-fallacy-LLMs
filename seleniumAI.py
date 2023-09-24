from selenium import webdriver

webdriver_path = "chromedriver.exe"


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')  # Use this line if you encounter sandbox-related issues

driver = webdriver.Chrome()

driver.get('https://chat.openai.com')

