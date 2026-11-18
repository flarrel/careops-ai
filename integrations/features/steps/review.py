from behave import given, then, when
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as conditions

WORKFLOW = '//section[@aria-label="Review workflow"]'
RESULT = (By.CSS_SELECTOR, 'section[aria-label="Review workflow"] div.result')


def click(context, label):
    button = (By.XPATH, f'{WORKFLOW}//button[normalize-space()="{label}"]')
    context.wait.until(conditions.element_to_be_clickable(button)).click()


@given('I request review of "{claim_id}"')
def request_review(context, claim_id):
    context.claim_id = claim_id
    click(context, "Synthetic review")
    context.browser.find_element(By.ID, "claim").send_keys(claim_id)
    click(context, "Request review")


@then("the application asks for permission without showing the fixture")
def pending(context):
    button = (By.XPATH, f'{WORKFLOW}//button[normalize-space()="Allow lookup"]')
    context.wait.until(conditions.visibility_of_element_located(button))
    hidden(context)


@when('I choose "{label}"')
def choose(context, label):
    click(context, label)


@then('the imported claim amount is "{expected}"')
def amount(context, expected):
    context.wait.until(conditions.text_to_be_present_in_element(RESULT, context.claim_id))
    result = context.browser.find_element(*RESULT)
    assert expected in result.text.replace("\u00a0", " "), result.text


@then('the checklist reports "{expected}"')
def checklist(context, expected):
    context.wait.until(conditions.text_to_be_present_in_element(RESULT, expected))


@then("the fixture is still hidden")
def hidden(context):
    section = context.browser.find_element(By.XPATH, WORKFLOW)
    assert context.claim_id not in section.text


@then('the run history shows "{expected}"')
def history(context, expected):
    first = (By.CSS_SELECTOR, 'section[aria-labelledby="history"] tbody tr:first-child')
    context.wait.until(conditions.text_to_be_present_in_element(first, expected))


@then('the application reports "{expected}"')
def invalid(context, expected):
    alert = (By.XPATH, f'{WORKFLOW}//*[@role="alert"]')
    context.wait.until(conditions.text_to_be_present_in_element(alert, expected))
    assert not context.browser.find_elements(By.XPATH, f'{WORKFLOW}//button[text()="Allow lookup"]')
