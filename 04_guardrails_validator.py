import json
import re
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

# --- Validator A: PII Detector ---

@register_validator(name="custom/pii_detector", data_type="string")
class PIIDetector(Validator):
    """
    Custom validator to detect PII: Email, US Phone, SSN, and Credit Cards.
    Replaces detected PII with [REDACTED] when on_fail=FIX.
    """
    def validate(self, value, metadata={}):
        pii_patterns = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
        }
        
        redacted_value = value
        found_pii = False
        
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, redacted_value):
                found_pii = True
                redacted_value = re.sub(pattern, "[REDACTED]", redacted_value)
        
        if found_pii:
            return FailResult(
                error_message="PII detected in output.",
                fix_value=redacted_value
            )
        
        return PassResult()

# --- Validator B: JSON Formatter ---

@register_validator(name="custom/json_formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates if string is JSON and attempts auto-repair.
    Repairs: strip markdown fences, fix single quotes, remove trailing commas.
    """
    def validate(self, value, metadata={}):
        original_value = value
        
        # Check if already valid
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Attempt Repair
        repaired = value.strip()
        
        # 1. Strip markdown fences
        repaired = re.sub(r"^```json\s*", "", repaired)
        repaired = re.sub(r"^```\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()
        
        # 2. Basic fixes: single quotes to double quotes, remove trailing commas
        repaired = repaired.replace("'", '"')
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        
        try:
            json.loads(repaired)
            # If repair succeeds, we use FailResult with fix_value to ensure override
            # because PassResult(value_override=...) behavior varies by version.
            return FailResult(
                error_message="JSON repaired.",
                fix_value=repaired
            )
        except json.JSONDecodeError:
            error_json = json.dumps({
                "error": "Invalid JSON format",
                "raw": original_value[:100] + "..." if len(original_value) > 100 else original_value
            })
            return FailResult(
                error_message="Failed to parse or repair JSON.",
                fix_value=error_json
            )

# --- Test Functions ---

def demo_pii():
    print("\n--- 🛡️ PII Detector Demo ---")
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    
    test_cases = [
        "Hello, my name is John Doe and I have no PII here.",
        "Contact me at john.doe@example.com for details.",
        "My phone number is 555-123-4567.",
        "Confidential SSN: 123-45-6789.",
        "Payment card: 1234-5678-9012-3456.",
        "Combined: email me at secret@agent.com or call 111-222-3333."
    ]
    
    for text in test_cases:
        result = guard.validate(text)
        is_redacted = result.validated_output != text
        status = "🚨 REDACTED" if is_redacted else "✅ CLEAN"
        print(f"[{status}] Input: {text}")
        print(f"         Output: {result.validated_output}")

def demo_json():
    print("\n--- 🛠️ JSON Formatter Demo ---")
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    
    test_cases = [
        '{"name": "Aetheria", "version": 2.4}',
        '```json\n{"name": "Aetheria", "status": "online"}\n```',
        "{'name': 'Flux-Lang', 'type': 'DSL'}",
        '{"items": [1, 2, 3,], "valid": false}',
        'This is definitely not JSON at all.'
    ]
    
    for text in test_cases:
        result = guard.validate(text)
        is_repaired = result.validated_output != text
        # Check if it contains error
        is_error = "Invalid JSON format" in str(result.validated_output)
        
        if is_error:
            status = "❌ ERROR"
        elif is_repaired:
            status = "🔧 REPAIRED"
        else:
            status = "✅ VALID"
            
        print(f"[{status}] Input: {text.replace(chr(10), ' ')}")
        print(f"         Output: {result.validated_output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "pii":
            demo_pii()
        elif sys.argv[1] == "json":
            demo_json()
    else:
        demo_pii()
        demo_json()
