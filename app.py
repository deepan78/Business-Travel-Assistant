
from flask import Flask, render_template, request

from graph import run_assistant

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    documents = []
    tool_result = None
    error = None
    active_section = None

    if request.method == "POST":
        action = request.form.get("action", "")

        try:
            if action == "ask_question":
                question = request.form.get("question", "").strip()

                if not question:
                    raise ValueError("Please enter a policy question.")

                result = run_assistant(
                    action="ask_question",
                    question=question,
                )

                answer = result.get("answer")
                sources = result.get("sources", [])
                documents = result.get("documents", [])
                active_section = "question"

            elif action == "estimate_reimbursement":
                grade = request.form.get("grade", "").strip().upper()
                city_category = (
                    request.form.get("city_category", "").strip().lower()
                )
                nights_text = request.form.get("nights", "").strip()
                charge_text = request.form.get("nightly_charge", "").strip()

                if not nights_text or not charge_text:
                    raise ValueError(
                        "Enter the number of nights and nightly charge."
                    )

                if not nights_text.isdigit():
                    raise ValueError(
                        "Number of nights must be a whole number."
                    )

                nights = int(nights_text)
                nightly_charge = float(charge_text)

                result = run_assistant(
                    action="estimate_reimbursement",
                    tool_inputs={
                        "grade": grade,
                        "city_category": city_category,
                        "nights": nights,
                        "nightly_charge": nightly_charge,
                    },
                )

                tool_result = result.get("tool_result")
                answer = result.get("answer")
                active_section = "tool"

            else:
                raise ValueError("Please select a valid action.")

        except (ValueError, TypeError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Travel assistant request failed")
            error = (
                "The request could not be completed. "
                "Check that Ollama is running and the project is configured."
            )

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        documents=documents,
        tool_result=tool_result,
        error=error,
        active_section=active_section,
    )


if __name__ == "__main__":
    app.run(debug=True)
