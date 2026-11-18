Feature: Review a synthetic claim imported from a spreadsheet

  Scenario: An operator allows a lookup
    Given I request review of "SYN-D-099"
    Then the application asks for permission without showing the fixture
    When I choose "Allow lookup"
    Then the imported claim amount is "IDR 1,234,567"
    And the checklist reports "Missing fixture attachment: invoice."
    And the run history shows "Needs human follow-up"

  Scenario: An operator declines a lookup
    Given I request review of "SYN-D-099"
    Then the application asks for permission without showing the fixture
    When I choose "Decline lookup"
    Then the checklist reports "Synthetic lookup declined."
    And the fixture is still hidden
    And the run history shows "Abstained"

  Scenario: An invalid identifier never reaches approval
    Given I request review of "bad-id"
    Then the application reports "Invalid request. Check claim_id."
