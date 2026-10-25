import json
import re

from src.dates import REFUND_DAYS, TODAY_DATE, parse_date, refund_overdue_days, return_deadline
from src.kb import find_products
from src.llm import chat
from src.rag import ALWAYS, K, describe_chunk, describe_product, fmt, mentions_old_date, refund_fact, strip_dates
from src.textnorm import digit_runs, normalize_speech, split_runs
from tools import create_ticket, lookup_order

HIGH_VALUE = 10000
CANCEL_SECTIONS = {
    "shipped": ["CAN-2", "REF-1"],
    "out_for_delivery": ["CAN-2", "REF-1"],
    "delivery_attempted": ["CAN-2", "REF-1"],
    "packed": ["CAN-1"],
    "placed": ["CAN-1"],
}
HIDDEN = {"found", "verified", "order_id", "customer_id", "delivery_pincode", "items", "customer_first_name", "is_plus"}
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
SCRIPT = re.compile(r"[\u0900-\u097F]")
LANGUAGES = {
    "en": "English",
    "hi": "Hindi written in Devanagari script",
    "hi-en": "Hinglish: Hindi written in Roman (English) letters, never Devanagari script",
}

ANALYSIS_SYSTEM = (
    "You analyse the latest customer message for a retail support agent. Output ONLY one JSON object with these keys:\n"
    "language: 'en', 'hi' (Hindi in Devanagari), 'hi-en' (Hinglish in Roman letters) or another ISO code;\n"
    "needs_order: true if answering needs details of the customer's own order (status, items, dates, refund) or the customer only says they need help with an order;\n"
    "wants_human: asks for a human agent or person;\n"
    "legal: mentions legal action, consumer court or a police complaint;\n"
    "unauthorized: reports an order, payment or account activity they did not make;\n"
    "safety: swelling battery, burning smell, sparks or injury;\n"
    "damage_claim: reports a damaged, defective or wrong item;\n"
    "repeat_contact: says they have contacted support about this several times;\n"
    "refund_query: asks about a refund;\n"
    "cancel_request: wants to cancel an order;\n"
    "item_hint: the product the customer refers to, else an empty string;\n"
    "search_query: a standalone English search query of at most 15 words using the conversation for context, keeping product names and payment methods, without dates or order numbers.\n"
    "All keys except language, item_hint and search_query are booleans."
)

AGENT_SYSTEM = (
    "You are ZentroMart's AI support assistant talking with a customer, possibly by phone. Follow the DIRECTIVE first, then use ORDER FACTS, CONTEXT, RULES and COMPUTED FACTS.\n"
    "- Write DECISION: answer, clarify or refuse on the first line, then a blank line, then the reply.\n"
    "- If FIRST_REPLY is yes, say you are ZentroMart's AI support assistant. Otherwise do not introduce yourself again.\n"
    "- Reply in the language and script given by REPLY LANGUAGE.\n"
    "- Reply in at most four short sentences, with one question at most. Include every number the policy gives for your answer (days, fees, amounts, timelines). Write dates with the month name and the year, and numbers as digits.\n"
    "- When documents disagree, follow KB-00.1. Use COMPUTED FACTS exactly as given and do no date arithmetic yourself.\n"
    "- For a return question, give the deadline date from ORDER FACTS and the pickup fee that applies to this customer, using the policy and whether they are a Plus member.\n"
    "- You cannot cancel orders, approve claims or issue refunds. For a cancellation request, explain what the policy allows and where the customer can do it. Never say something was done unless the DIRECTIVE says so.\n"
    "- If the customer demands compensation or an exception the policy does not allow, apologise and politely refuse, without promising anything.\n"
    "- Never read out a phone number, email, address or pincode, and never reveal what a verification value should be."
)


def human(value):
    return fmt(parse_date(value))


def render_value(key, value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str) and ISO.match(value):
        if key == "timeline":
            return re.sub(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}", lambda match: human(match.group(1)), value)
        return human(value)
    if key == "timeline":
        return re.sub(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}", lambda match: human(match.group(1)), str(value))
    return str(value)


def refund_timing(method):
    if method == "cod":
        return "Cash on delivery refunds: to ZentroMart Wallet instantly (within 2 hours), or by bank transfer in 5 to 7 business days after bank details are verified."
    key = {"upi": "upi", "card": "card", "netbanking": "netbanking", "wallet": "wallet"}.get(method)
    if not key:
        return None
    low, high = REFUND_DAYS[key]
    if high == 0:
        return f"Refunds to {method} are instant, within 2 hours."
    return f"Refunds to {method} take {low} to {high} business days (Sundays and warehouse holidays not counted)."


def return_progress(order):
    if not order.get("return_sku"):
        return None
    steps = []
    if order.get("return_requested_on"):
        steps.append(f"requested on {human(order['return_requested_on'])}")
    if order.get("return_pickup_on"):
        steps.append(f"picked up on {human(order['return_pickup_on'])}")
    else:
        steps.append("pickup not done yet")
    if order.get("return_qc_passed_on"):
        steps.append(f"quality check (QC) passed on {human(order['return_qc_passed_on'])}")
    elif order.get("return_pickup_on"):
        steps.append("quality check (QC) is still pending, so the refund has not been initiated yet; it starts after the item passes QC")
    if order.get("return_refund_initiated_on"):
        text = f"refund of INR {order.get('return_refund_amount_inr')} initiated on {human(order['return_refund_initiated_on'])} to {order.get('return_refund_method')}"
        if order.get("return_rrn"):
            text += f", reference number (RRN) {order['return_rrn']}"
        steps.append(text)
    if order.get("return_refund_completed_on"):
        steps.append(f"refund completed on {human(order['return_refund_completed_on'])}")
    return "return progress: " + "; ".join(steps)


def describe_order(order):
    lines = ["verification: the customer has been verified, so order details can be shared"]
    name = order.get("customer_first_name")
    lines.append(f"customer first name: {name}; ZentroMart Plus member: {'yes' if order.get('is_plus') else 'no'}")
    for key, value in order.items():
        if key in HIDDEN or value is None or value == "":
            continue
        lines.append(f"{key}: {render_value(key, value)}")
    delivered = order.get("delivered_on")
    for item in order.get("items", []):
        line = (
            f"item {item['line_no']}: {item['title']} (sku {item['sku']}), qty {item['qty']}, "
            f"price INR {item['unit_price_inr']}, category {item['return_category']}"
        )
        if delivered:
            deadline = return_deadline(item["return_category"], delivered, order["placed_at"])
            if deadline is None:
                line += "; this category cannot be returned"
            else:
                state = "still open" if deadline >= TODAY_DATE else "already closed"
                line += f"; return window closes on {fmt(deadline)} ({state} as of {fmt(TODAY_DATE)})"
        lines.append(line)
    progress = return_progress(order)
    if progress:
        lines.append(progress)
    timing = refund_timing(order.get("payment_method"))
    if timing:
        lines.append(f"once a refund is initiated, timing for this order's payment method: {timing}")
    return "\n".join(lines)


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def split_decision(raw, default="answer"):
    match = re.match(r"\s*[*_#> ]*DECISION[*_: ]*(answer|clarify|refuse|escalate)[*_ ]*\n+(.*)", raw, re.S | re.I)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return default, re.sub(r"^[*_#> ]*DECISION[^\n]*\n+", "", raw, flags=re.I).strip()


def title_tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Agent:
    def __init__(self, rag):
        self.rag = rag
        self.history = []
        self.order_id = None
        self.order = None
        self.attempts = 0
        self.locked = False
        self.turn = 0
        self.ticketed = False
        self.language = "en"
        self.tools = []

    def respond(self, text):
        self.turn += 1
        self.tools = []
        runs = digit_runs(normalize_speech(text))
        plan = None
        if self.order is None and not self.locked:
            plan = self.order_flow(runs)
        if plan is None:
            analysis = self.analyse(text)
            plan = self.decide(text, analysis)
        return self.compose(text, plan)

    def order_flow(self, runs):
        order_id, proof = split_runs(runs)
        fresh = False
        if self.order_id is None:
            if not order_id:
                return None
            self.order_id = order_id
            self.tools.append(f"lookup_order(order_id={order_id})")
            lookup_order(order_id)
            fresh = True
        if proof is None:
            if fresh:
                return {
                    "decision": "clarify",
                    "directive": "The customer gave the order ID. Do not reveal anything about the order. Ask for ONE of: the last 4 digits of the registered phone number, or the delivery pincode.",
                }
            return None
        self.tools.append(f"lookup_order(order_id={self.order_id}, verification_value={proof})")
        result = lookup_order(self.order_id, proof)
        if result.get("verified"):
            self.order = result
            return {
                "decision": "answer",
                "directive": "Verification succeeded. Say so briefly, then tell the customer the current status of the order and the relevant date (the promised delivery date if it has not been delivered, otherwise the delivery date), and ask how you can help. Do not introduce yourself again.",
                "order": True,
            }
        self.attempts += 1
        if self.attempts >= 2:
            self.locked = True
            return {
                "decision": "refuse",
                "directive": "Verification failed twice. Share no order details at all. Say you cannot share order details because the values do not match the registered details, and offer help with general policy questions, or suggest contacting support from the registered phone number. Never reveal the expected value.",
            }
        return {
            "decision": "clarify",
            "directive": "The value did not match. Say it does not match and ask the customer to try again, once, with the last 4 digits of the registered phone number or the delivery pincode. Share nothing about the order.",
        }

    def analyse(self, text):
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-4:])
        message = f"CONVERSATION SO FAR:\n{recent or 'none'}\n\nLATEST CUSTOMER MESSAGE:\n{text}"
        data = parse_json(chat(ANALYSIS_SYSTEM, [{"role": "user", "content": message}], temperature=0, max_tokens=800))
        for key in ["needs_order", "wants_human", "legal", "unauthorized", "safety", "damage_claim", "repeat_contact", "refund_query", "cancel_request"]:
            data[key] = bool(data.get(key, False))
        label = data.get("language")
        if label in LANGUAGES:
            if label == "hi" and not SCRIPT.search(text):
                label = "hi-en"
            self.language = label
        data["item_hint"] = str(data.get("item_hint") or "")
        data["search_query"] = str(data.get("search_query") or "")
        return data

    def claim_value(self, text, analysis):
        prices = []
        hint = title_tokens(analysis["item_hint"])
        if self.order:
            for item in self.order.get("items", []):
                words = title_tokens(item["title"])
                if (hint and len(hint & words) >= max(1, len(hint) // 2)) or item["sku"] == self.order.get("damage_sku"):
                    prices.append(item["unit_price_inr"])
        for product in find_products(text) + find_products(analysis["item_hint"]):
            prices.append(int(product["price_inr"]))
        return max(prices) if prices else 0

    def overdue_refund(self):
        order = self.order
        start, method = order.get("return_refund_initiated_on"), order.get("return_refund_method")
        if start and method and not order.get("return_refund_completed_on"):
            days = refund_overdue_days(start, method)
            if days > 3:
                return {"rrn": order.get("return_rrn"), "days": days}
        return None

    def escalation(self, text, analysis):
        if analysis["wants_human"]:
            return "human_requested", "customer asked for a human agent", ""
        if analysis["legal"]:
            return "legal_threat", "customer mentioned legal action, a consumer court or a police complaint", ""
        if analysis["unauthorized"]:
            return "unauthorized_activity", "customer reports activity they did not make", ""
        if analysis["safety"]:
            return "safety_risk", "product safety risk reported", ""
        if analysis["damage_claim"] and self.claim_value(text, analysis) > HIGH_VALUE:
            return "high_value_damage", "damaged, defective or wrong item worth more than INR 10000", "This claim is handled by the Priority Resolution team. Do not approve or promise anything."
        if self.order:
            if self.order.get("contacts_last_7_days", 0) >= 2 and (analysis["repeat_contact"] or analysis["needs_order"]):
                return "repeat_contact", "third or later contact about the same issue within 7 days", "Acknowledge that the customer has contacted us several times."
            if analysis["refund_query"]:
                overdue = self.overdue_refund()
                if overdue:
                    return (
                        "refund_overdue",
                        "refund is more than 3 business days past its maximum timeline",
                        f"Tell the customer the refund is overdue and share the refund reference number (RRN) {overdue['rrn']}. Do not say when it will arrive.",
                    )
        return None

    def ticket_summary(self, text, intent, reason):
        told = " | ".join(m["content"] for m in self.history if m["role"] == "user")
        told = (told + " | " + text).strip(" |")[-400:]
        checked = (
            f"caller verified, order status {self.order.get('status')}, contacts in last 7 days {self.order.get('contacts_last_7_days')}"
            if self.order
            else "caller not verified, no order details checked"
        )
        return f"Customer said: {told}. Detected intent: {intent}. Order ID: {self.order_id or 'none'}. Checked: {checked}. Escalation reason: {reason}."

    def escalate(self, text, found):
        intent, reason, extra = found
        facts = [reason]
        if self.ticketed:
            directive = "A support ticket was already created in this conversation. Say a human colleague will contact the customer. Promise no outcome or time. " + extra
            return {"decision": "escalate", "directive": directive, "order": bool(self.order), "facts": facts}
        summary = self.ticket_summary(text, intent, reason)
        self.tools.append("create_ticket()")
        result = create_ticket(summary, order_id=self.order_id, reason=reason, intent=intent)
        if result.get("created"):
            self.ticketed = True
            directive = (
                "A support ticket has been created. Say a human colleague will contact the customer about this, and use the word ticket. "
                "Promise no outcome, amount, approval or time. " + extra
            )
        else:
            directive = "The ticket could NOT be created. Apologise, say you could not hand this over right now, and ask the customer to try again shortly. Do not say a ticket exists."
        return {"decision": "escalate", "directive": directive, "order": bool(self.order), "facts": facts}

    def decide(self, text, analysis):
        found = self.escalation(text, analysis)
        if found:
            return self.escalate(text, found)
        if analysis["needs_order"] and not self.order:
            if self.locked:
                return {
                    "decision": "refuse",
                    "directive": "Order details cannot be shared because verification failed. Politely say so and offer help with general policy questions.",
                }
            if self.order_id is None:
                return {
                    "decision": "clarify",
                    "directive": "The customer needs help with their order. Ask for the order ID, as the only question.",
                }
            return {
                "decision": "clarify",
                "directive": "Before sharing anything about the order, ask for ONE of: the last 4 digits of the registered phone number, or the delivery pincode. Share nothing about the order.",
            }
        return self.answer_plan(text, analysis)

    def cross_references(self, hits, limit=2):
        present = {chunk["id"] for chunk in hits}
        extra = []
        for chunk in hits:
            for ref in re.findall(r"\b[A-Z]{2,5}-\d+\b", chunk["text"]):
                other = self.rag.by_id.get(ref)
                if other and ref not in present and other["status"] != "superseded":
                    extra.append(other)
                    present.add(ref)
                    if len(extra) == limit:
                        return extra
        return extra

    def has_old_delivery(self):
        if not self.order:
            return False
        delivered = self.order.get("delivered_on")
        return bool(delivered) and parse_date(delivered) < parse_date("2026-08-01")

    def answer_plan(self, text, analysis):
        queries = [analysis["search_query"] or text, strip_dates(text) or text]
        old = mentions_old_date(text) or self.has_old_delivery()
        hits = [chunk for chunk, _ in self.rag.retriever.search(queries, K, include_superseded=old)]
        shipped_cancel = bool(
            self.order
            and analysis["cancel_request"]
            and self.order.get("status") in ("shipped", "out_for_delivery", "delivery_attempted")
        )
        if self.order and analysis["cancel_request"]:
            present = {chunk["id"] for chunk in hits}
            hits = hits + [
                self.rag.by_id[i]
                for i in CANCEL_SECTIONS.get(self.order.get("status"), [])
                if i in self.rag.by_id and i not in present
            ]
        hits = hits + self.cross_references(hits)
        hit_ids = [chunk["id"] for chunk in hits]
        rules = [self.rag.by_id[i] for i in ALWAYS if i in self.rag.by_id and i not in hit_ids]
        products = find_products(text)
        facts = [fact for fact in [refund_fact(text)] if fact]
        context = "\n\n".join([describe_chunk(chunk) for chunk in hits] + [describe_product(p) for p in products])
        return {
            "decision": None,
            "directive": "Answer the customer's latest message."
            + (
                " The order has already shipped: explain the options the policy gives and, for a refused or returned parcel, state the refund timeline from the policy in days."
                if shipped_cancel
                else ""
            ),
            "order": bool(self.order),
            "context": context,
            "rules": "\n\n".join(describe_chunk(chunk) for chunk in rules),
            "facts": facts,
            "retrieved": hit_ids + [chunk["id"] for chunk in rules] + ["products.csv:" + p["sku"] for p in products],
        }

    def compose(self, text, plan):
        if SCRIPT.search(text):
            self.language = "hi"
        directive = plan["directive"]
        if self.turn == 1:
            directive = "This is your first reply: begin by saying you are ZentroMart's AI support assistant, using the words AI assistant. " + directive
        parts = [
            f"FIRST_REPLY: {'yes' if self.turn == 1 else 'no'}",
            f"REPLY LANGUAGE: {LANGUAGES.get(self.language, 'the same language and script as the customer')}",
            f"DIRECTIVE: {directive}",
        ]
        if plan.get("order") and self.order:
            parts.append("ORDER FACTS:\n" + describe_order(self.order))
        if plan.get("context"):
            parts.append("CONTEXT:\n" + plan["context"])
        if plan.get("rules"):
            parts.append("RULES:\n" + plan["rules"])
        facts = plan.get("facts") or []
        computed = [fact for fact in facts if fact.startswith("Refund")]
        if computed:
            parts.append("COMPUTED FACTS:\n" + "\n".join(computed))
        parts.append("CUSTOMER MESSAGE:\n" + text)
        raw = chat(AGENT_SYSTEM, self.history[-8:] + [{"role": "user", "content": "\n\n".join(parts)}])
        said, reply = split_decision(raw)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "decision": plan.get("decision") or said,
            "tools": list(self.tools),
            "retrieved": plan.get("retrieved", []),
            "facts": facts,
            "language": self.language,
        }