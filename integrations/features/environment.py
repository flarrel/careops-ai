"""Run the feature against the built web client in headless Chrome."""

import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.utils import is_url_connectable
from selenium.webdriver.support.ui import WebDriverWait


class LoopbackChromeService(Service):
    @property
    def service_url(self):
        return f"http://127.0.0.1:{self.port}"

    def is_connectable(self):
        return is_url_connectable(self.port, host="127.0.0.1")


def before_scenario(context, _scenario):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--remote-debugging-pipe")
    options.add_argument("--window-size=1200,900")
    if os.environ.get("CHROME_BINARY"):
        options.binary_location = os.environ["CHROME_BINARY"]
    context.browser = webdriver.Chrome(
        options=options,
        service=LoopbackChromeService(executable_path=os.environ.get("CHROMEDRIVER")),
    )
    context.add_cleanup(context.browser.quit)
    context.wait = WebDriverWait(context.browser, 20)
    context.browser.get(os.environ.get("CAREOPS_WEB_URL", "http://127.0.0.1:3000"))
