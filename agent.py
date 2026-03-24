from langchain_community.llms import Ollama

llm = Ollama(model="mistral")


# ------------------ TOOL ------------------ #
def check_user_history(user_id):
    return "User has typical low to moderate transaction behavior."


# ------------------ AGENT LOGIC ------------------ #
def fraud_agent(transaction_data, model_result):

    try:
        user_id = transaction_data.get("user_id", "unknown")

        # 🧠 STEP 1: Decide tool usage
        tool_decision_prompt = f"""
You are an intelligent fraud detection agent.

Transaction risk score: {model_result['risk_score']}

Should you check user transaction history?

Answer ONLY: YES or NO
"""

        tool_decision = llm.invoke(tool_decision_prompt)

        if not tool_decision:
            tool_decision = "NO"

        tool_decision = tool_decision.strip().upper()

        # 🤖 STEP 2: Tool execution
        if "YES" in tool_decision:
            history = check_user_history(user_id)
        else:
            history = "User history not checked"

        # 🧠 STEP 3: FINAL DECISION
        final_prompt = f"""
You are an AI Fraud Detection Agent.

IMPORTANT RULES:
- Use ONLY given data
- Do NOT assume user habits
- Do NOT invent behavioral patterns

Transaction Details:
- Amount: {transaction_data['TransactionAmt']}
- Distance: {model_result['distance']} km
- Risk Score: {model_result['risk_score']}
- User History: {history}

Analyze:
1. Risk level
2. Behavior vs available data
3. Final decision

Return STRICT format:

Decision: Fraud / Suspicious / Safe

Reason:
Clear explanation based ONLY on input

Action:
Accept / Flag / Block
"""

        response = llm.invoke(final_prompt)

        if not response or response.strip() == "":
            return fallback_response(model_result)

        return response

    except Exception:
        return fallback_response(model_result)


# ------------------ PARSER ------------------ #
def parse_agent_output(output: str):
    decision = "Safe"
    action = "Accept"

    try:
        lines = output.split("\n")

        for line in lines:
            if "Decision:" in line:
                decision = line.split("Decision:")[1].strip()

            if "Action:" in line:
                action = line.split("Action:")[1].strip()

    except:
        pass

    if decision not in ["Fraud", "Suspicious", "Safe"]:
        decision = "Suspicious"

    if action not in ["Accept", "Flag", "Block"]:
        action = "Flag"

    return decision, action


# ------------------ FALLBACK ------------------ #
def fallback_response(model_result):

    risk = model_result["risk_score"]
    distance = model_result["distance"]

    if risk > 70:
        decision = "Fraud"
        action = "Block"
        reason = f"High risk transaction ({risk:.2f}%). Large deviation in behavior or distance ({distance} km)."

    elif risk > 40:
        decision = "Suspicious"
        action = "Flag"
        reason = f"Moderate risk ({risk:.2f}%). Some unusual signals detected."

    else:
        decision = "Safe"
        action = "Accept"
        reason = f"Low risk ({risk:.2f}%). Behavior appears normal."

    return f"""
Decision: {decision}

Reason:
{reason}

Action:
{action}
"""