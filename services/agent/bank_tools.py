"""Tools and system prompt for the banking voice agent (shared by the agent and benchmarks)."""

SYSTEM_PROMPT = (
    "Sen Freya Bank'ın telefon asistanı Leyla'sın. Müşteriyle telefonda Türkçe konuşuyorsun. "
    "Cevapların tek ve kısa bir cümle olsun, en fazla on beş kelime. "
    "Rakam kullanma, sayıları yazıyla yaz. Emoji, madde işareti veya markdown kullanma. "
    "Kart kaybı veya çalınması için block_card, ekstre ve borç soruları için get_statement, "
    "müşteri insan temsilci isterse ya da sen çözemezsen transfer_to_agent aracını kullan."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "block_card",
            "description": "Kayıp veya çalıntı kartı geçici olarak kullanıma kapatır.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "enum": ["lost", "stolen"]}}, "required": ["reason"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statement",
            "description": "Kredi kartı ekstresi: son ödeme tarihi, asgari tutar, toplam borç.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_agent",
            "description": "Görüşmeyi insan müşteri temsilcisine aktarır.",
            "parameters": {"type": "object", "properties": {
                "summary": {"type": "string", "description": "Temsilci için tek cümlelik özet"}},
                "required": ["summary"]},
        },
    },
]


def run_tool(name: str, args: dict) -> dict:
    """Mock core-banking backend."""
    if name == "block_card":
        return {"status": "blocked", "new_card_days": 3}
    if name == "get_statement":
        return {"due_date": "15 Ekim", "minimum_payment_try": 2400, "total_debt_try": 18750}
    if name == "transfer_to_agent":
        return {"status": "transferring"}
    return {"error": f"unknown tool {name}"}


# ---------------------------------------------------------------------------
# Structured-output variant. Small models are unreliable at free-form tool calls
# (they sometimes *say* the card is blocked without calling the tool). Instead the
# model must return JSON matching this schema (enforced by constrained decoding),
# and the agent -- not the model -- executes the action for the chosen intent.
# ---------------------------------------------------------------------------
INTENTS = ["block_card", "get_statement", "transfer_to_agent", "smalltalk", "unknown"]

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "reply": {"type": "string"},
    },
    "required": ["intent", "reply"],
}

INTENT_PROMPT = (
    "Sen Freya Bank'ın telefon asistanı Leyla'sın. Müşterinin son cümlesini sınıflandır ve kısa bir cevap yaz.\n"
    "intent değerleri:\n"
    "- block_card: kart kayboldu, çalındı, kartı kapatmak istiyor\n"
    "- get_statement: borç, ekstre, asgari ödeme, son ödeme tarihi\n"
    "- transfer_to_agent: insan, temsilci, yetkili ile görüşmek istiyor\n"
    "- smalltalk: selamlaşma, teşekkür\n"
    "- unknown: diğer her şey\n"
    "reply: tek cümle, en fazla on iki kelime, rakam yok. İşlem yapıldı deme; "
    "block_card, get_statement ve transfer_to_agent için sadece 'Hemen kontrol ediyorum.' gibi kısa bir bekleme cümlesi yaz."
)
