"""Banking plugin: the tools the voice agent can execute, backed by a mock core-banking API.

Every tool returns a dict with a pre-written Turkish sentence in "say", so the words
spoken after an action come from code, not from the model.
"""
from tools import ToolRegistry

bank = ToolRegistry()


@bank.tool(
    "Kayıp veya çalıntı kartı geçici olarak kullanıma kapatır. Örnek: kartımı kaybettim, kartım çalındı, kartı kapatın.",
    parameters={"reason": {"type": "string", "enum": ["lost", "stolen"]}},
)
def block_card(reason: str = "lost") -> dict:
    return {"status": "blocked", "reason": reason,
            "say": "Kartınızı güvenliğiniz için kullanıma kapattım. Yeni kartınız üç gün içinde adresinize ulaşacak."}


@bank.tool("Kredi kartı ekstresi: borç, asgari ödeme tutarı, son ödeme tarihi. Örnek: ne kadar borcum var, son ödeme tarihim ne zaman.")
def get_statement() -> dict:
    return {"due_date": "15 Ekim", "minimum_payment_try": 2400, "total_debt_try": 18750,
            "say": "Son ödeme tarihiniz on beş ekim. Asgari tutar iki bin dört yüz lira."}


@bank.tool(
    "Görüşmeyi insan müşteri temsilcisine aktarır.",
    parameters={"summary": {"type": "string", "description": "Temsilci için tek cümlelik özet"}},
)
def transfer_to_agent(summary: str = "") -> dict:
    return {"status": "transferring", "summary": summary,
            "say": "Sizi müşteri temsilcimize aktarıyorum, lütfen hattan ayrılmayın."}


def fixed_phrases() -> list[str]:
    """Sentences the tools can speak, for pre-rendering at start-up."""
    return [block_card()["say"], get_statement()["say"], transfer_to_agent()["say"]]


PERSONA = (
    "Sen Freya Bank'ın telefon asistanı Leyla'sın. Müşteriyle telefonda Türkçe konuşuyorsun. "
    "Cevapların tek ve kısa bir cümle olsun, en fazla on iki kelime. "
    "Rakam kullanma, sayıları yazıyla yaz. Emoji, madde işareti veya markdown kullanma. "
    "Saat, faiz, ücret, şube gibi bilmediğin hiçbir bilgiyi uydurma; bunun için temsilciye bağlanabileceğini söyle."
)

# --- kept for bench/llm_bench.py and bench/llm_intent_bench.py -------------------------
SYSTEM_PROMPT = PERSONA + (
    " Kart kaybı veya çalınması için block_card, ekstre ve borç soruları için get_statement, "
    "müşteri insan temsilci isterse ya da sen çözemezsen transfer_to_agent aracını kullan."
)
TOOLS = bank.function_schemas()
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["block_card", "get_statement", "transfer_to_agent", "smalltalk", "unknown"]},
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
