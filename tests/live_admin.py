"""Exercise the actual REST API through Streamlit AppTest from the repository root."""

from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file("app/admin.py", default_timeout=60).run()
    assert not app.exception

    app.text_area[0].set_value("Di bagian mana rincian pedoman INA-CBG 2021 ditetapkan?")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state.answer["status"] == "answer"
    assert app.get("link_button")

    app.text_input[0].set_value("SYN-D-001")
    app.number_input[0].set_value(400000)
    next(button for button in app.button if button.label == "Request review").click().run()
    assert set(app.session_state.pending) == {"run_id", "status"}
    next(button for button in app.button if button.label == "Allow lookup").click().run()
    assert not app.exception and not app.error
    assert app.session_state.review["lookup"]["claim_id"] == "SYN-D-001"
    assert app.session_state.review["result"]["status"] == "escalate"
    assert len(app.dataframe) == 1
    print("Streamlit admin passed cited policy, approved review, and history against FastAPI")


if __name__ == "__main__":
    main()
