"""
Unit tests for EvalMatch Career Copilot core functions.

Run with: python -m pytest test_app.py -v
or: python test_app.py
"""

import sys
from app import extract_text, select_variant, extract_slots, _join_tokens, merge_sampling, TEMPLATES


def test_join_tokens_string_tokens():
    """Test _join_tokens with plain string tokens."""
    tokens = ["Hello", " ", "world", "!"]
    assert _join_tokens(tokens) == "Hello world!"


def test_join_tokens_dict_tokens():
    """Test _join_tokens with dict tokens containing 'text' key."""
    tokens = [
        {"text": "Hello", "id": 1},
        {"text": " ", "id": 2},
        {"text": "world", "id": 3}
    ]
    assert _join_tokens(tokens) == "Hello world"


def test_join_tokens_mixed():
    """Test _join_tokens with mixed string and dict tokens."""
    tokens = [
        "Hello",
        {"text": " world"},
        "!",
        {"text": ""}
    ]
    assert _join_tokens(tokens) == "Hello world!"


def test_join_tokens_empty():
    """Test _join_tokens with empty list."""
    assert _join_tokens([]) == ""


def test_merge_sampling_min_tokens():
    """Test merge_sampling with min_tokens ensures max_tokens is at least min_tokens."""
    base = {"max_tokens": 200, "temperature": 0.3}
    template = {"min_tokens": 260}
    result = merge_sampling(base, template)
    assert result["max_tokens"] == 260
    assert result["temperature"] == 0.3


def test_merge_sampling_direct_override():
    """Test merge_sampling with direct override (repetition_penalty)."""
    base = {"max_tokens": 200, "temperature": 0.3, "repetition_penalty": 1.0}
    template = {"repetition_penalty": 1.05}
    result = merge_sampling(base, template)
    assert result["repetition_penalty"] == 1.05
    assert result["max_tokens"] == 200
    assert result["temperature"] == 0.3


def test_merge_sampling_multiple_overrides():
    """Test merge_sampling with multiple overrides including min_tokens."""
    base = {"max_tokens": 200, "temperature": 0.3, "top_p": 0.9}
    template = {"min_tokens": 250, "temperature": 0.5, "repetition_penalty": 1.1}
    result = merge_sampling(base, template)
    assert result["max_tokens"] == 250  # Updated from min_tokens
    assert result["temperature"] == 0.5  # Overridden
    assert result["top_p"] == 0.9  # Preserved
    assert result["repetition_penalty"] == 1.1  # Added


def test_merge_sampling_no_mutation():
    """Test merge_sampling does not mutate input dicts."""
    base = {"max_tokens": 200, "temperature": 0.3}
    template = {"temperature": 0.5}
    result = merge_sampling(base, template)
    assert base["temperature"] == 0.3  # Original unchanged
    assert result["temperature"] == 0.5


def test_extract_text_handler_format():
    """Test extract_text with handler endpoint format (dict with choices[0].text)."""
    payload = {
        "output": {
            "choices": [
                {"text": "Response from handler"}
            ]
        }
    }
    assert extract_text(payload) == "Response from handler"


def test_extract_text_handler_format_tokens():
    """Test extract_text with handler endpoint format (dict with choices[0].tokens)."""
    payload = {
        "output": {
            "choices": [
                {"tokens": ["Token1", " ", "Token2"]}
            ]
        }
    }
    assert extract_text(payload) == "Token1 Token2"


def test_extract_text_raw_vllm_format():
    """Test extract_text with raw vLLM format (list with choices[0].tokens)."""
    payload = {
        "output": [
            {
                "choices": [
                    {"tokens": ["Raw", " ", "vLLM", " ", "response"]}
                ]
            }
        ]
    }
    assert extract_text(payload) == "Raw vLLM response"


def test_extract_text_raw_vllm_text():
    """Test extract_text with raw vLLM format (list with choices[0].text)."""
    payload = {
        "output": [
            {
                "choices": [
                    {"text": "Raw vLLM text response"}
                ]
            }
        ]
    }
    assert extract_text(payload) == "Raw vLLM text response"


def test_extract_text_string_output():
    """Test extract_text with plain string output."""
    payload = {"output": "Plain string response"}
    assert extract_text(payload) == "Plain string response"


def test_extract_text_root_text():
    """Test extract_text with text at payload root."""
    payload = {"text": "Root text response"}
    assert extract_text(payload) == "Root text response"


def test_extract_text_dict_tokens_objects():
    """Test extract_text with dict tokens (object tokens from some vLLM configs)."""
    payload = {
        "output": {
            "choices": [
                {
                    "tokens": [
                        {"text": "Object", "logprob": -0.5},
                        {"text": " token", "logprob": -0.3},
                        {"text": " test", "logprob": -0.2}
                    ]
                }
            ]
        }
    }
    assert extract_text(payload) == "Object token test"


def test_select_variant_resume_senior_lead():
    """Test select_variant detects senior_lead variant for resume guidance."""
    slots = {"role": "Backend Engineer", "stack": "python", "location": None}
    variants = select_variant("resume_guidance", "Optimize my resume for Senior Backend Engineer", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "senior_lead"


def test_select_variant_resume_technical_depth():
    """Test select_variant detects technical_depth variant for resume guidance."""
    slots = {"role": "Backend Engineer", "stack": "python", "location": None}
    variants = select_variant("resume_guidance", "Optimize my resume for architect with system design experience", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "technical_depth"


def test_select_variant_jd_dual_variant():
    """Test select_variant selects BOTH tech_deep_dive and remote_distributed for JD."""
    slots = {"role": "DevOps Engineer", "stack": "kubernetes, terraform", "location": "Remote"}
    variants = select_variant("job_description", "Create JD for remote DevOps engineer with Kubernetes and SOC 2 compliance", slots)
    assert len(variants) == 2
    variant_names = [v["name"] for v in variants]
    assert "tech_deep_dive" in variant_names
    assert "remote_distributed" in variant_names


def test_select_variant_jd_tech_only():
    """Test select_variant selects only tech_deep_dive when no remote mentioned."""
    slots = {"role": "Backend Engineer", "stack": "python, kafka", "location": None}
    variants = select_variant("job_description", "Create JD for Backend Engineer with Python and Kafka", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "tech_deep_dive"


def test_select_variant_jd_remote_only():
    """Test select_variant selects only remote_distributed when no stack."""
    slots = {"role": "Backend Engineer", "stack": None, "location": "Remote"}
    variants = select_variant("job_description", "Create JD for remote Backend Engineer", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "remote_distributed"


def test_select_variant_recruiting_streaming():
    """Test select_variant detects streaming_aware variant."""
    slots = {"role": "Backend Engineer", "stack": "kafka", "location": None}
    variants = select_variant("recruiting_strategy", "Strategy to hire Kafka streaming engineers", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "streaming_aware"


def test_select_variant_recruiting_senior_exec():
    """Test select_variant detects senior_executive variant (not streaming)."""
    slots = {"role": "CTO", "stack": None, "location": None}
    variants = select_variant("recruiting_strategy", "Strategy to hire senior executive CTO", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "senior_executive"


def test_select_variant_match_quick_screen():
    """Test select_variant detects quick_screen variant with word boundaries."""
    slots = {"role": None, "stack": None, "location": None}
    variants = select_variant("job_resume_match", "Quick screening of candidate", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "quick_screen"


def test_select_variant_ats_role_specific():
    """Test select_variant detects role_specific variant when role is present."""
    slots = {"role": "DevOps Engineer", "stack": "kubernetes", "location": None}
    variants = select_variant("ats_keywords", "ATS keywords for DevOps Engineer", slots)
    assert len(variants) == 1
    assert variants[0]["name"] == "role_specific"


def test_select_variant_no_match():
    """Test select_variant returns empty list when no variants match."""
    slots = {"role": None, "stack": None, "location": None}
    variants = select_variant("resume_guidance", "Optimize my resume", slots)
    assert variants == []


def test_extract_slots_role():
    """Test extract_slots detects role."""
    slots = extract_slots("Create resume for Senior Backend Engineer")
    assert slots["role"] == "Backend Engineer"


def test_extract_slots_stack_multiple():
    """Test extract_slots detects multiple stacks (top 3)."""
    slots = extract_slots("Looking for Python, Kafka, Kubernetes, Docker, AWS, Terraform engineer")
    stacks = slots["stack"].split(", ")
    assert len(stacks) == 3
    assert "python" in stacks
    assert "kafka" in stacks
    assert "kubernetes" in stacks


def test_extract_slots_go_with_boundaries():
    """Test extract_slots detects 'go' with word boundaries (not 'going', 'goal')."""
    slots = extract_slots("I'm going to hire a Go developer for the goal")
    assert slots["stack"] == "go"


def test_extract_slots_golang():
    """Test extract_slots detects 'golang' separately."""
    slots = extract_slots("Looking for Golang expert")
    assert slots["stack"] == "golang"


def test_extract_slots_go_and_golang():
    """Test extract_slots includes both if both present."""
    slots = extract_slots("We need Go and Golang developers")
    stacks = slots["stack"].split(", ")
    assert "golang" in stacks
    assert "go" in stacks


def test_extract_slots_location_sf():
    """Test extract_slots detects SF with word boundaries."""
    slots = extract_slots("Hiring in SF area")
    assert slots["location"] == "San Francisco"


def test_extract_slots_location_nyc():
    """Test extract_slots detects NYC with word boundaries."""
    slots = extract_slots("Looking for engineers in NYC")
    assert slots["location"] == "New York"


def test_extract_slots_location_remote():
    """Test extract_slots detects remote location."""
    slots = extract_slots("Remote position for distributed team")
    assert slots["location"] == "Remote"


def test_extract_slots_no_false_positives():
    """Test extract_slots avoids false positives (e.g., 'go' in 'going')."""
    slots = extract_slots("I am going to the conference")
    assert slots["stack"] is None  # Should not detect 'go'


def test_extract_slots_combined():
    """Test extract_slots with role, stack, and location."""
    slots = extract_slots("Remote Senior DevOps Engineer position with Kubernetes and Python in San Francisco")
    assert slots["role"] == "DevOps Engineer"
    assert "kubernetes" in slots["stack"]
    assert "python" in slots["stack"]
    # Remote takes precedence over SF
    assert slots["location"] == "Remote"


def test_templates_have_seed_flag():
    """Test all templates have seed flag defined."""
    for domain, template in TEMPLATES.items():
        assert "seed" in template, f"Template {domain} missing 'seed' flag"
        assert isinstance(template["seed"], bool), f"Template {domain} 'seed' must be bool"


def test_templates_job_resume_match_has_sampling():
    """Test job_resume_match template has sampling override."""
    assert "sampling" in TEMPLATES["job_resume_match"]
    assert "min_tokens" in TEMPLATES["job_resume_match"]["sampling"]
    assert TEMPLATES["job_resume_match"]["sampling"]["min_tokens"] == 260


def run_tests():
    """Run all tests manually (for environments without pytest)."""
    test_functions = [
        (test_join_tokens_string_tokens, "join_tokens with string tokens"),
        (test_join_tokens_dict_tokens, "join_tokens with dict tokens"),
        (test_join_tokens_mixed, "join_tokens with mixed tokens"),
        (test_join_tokens_empty, "join_tokens with empty list"),
        (test_merge_sampling_min_tokens, "merge_sampling with min_tokens"),
        (test_merge_sampling_direct_override, "merge_sampling direct override"),
        (test_merge_sampling_multiple_overrides, "merge_sampling multiple overrides"),
        (test_merge_sampling_no_mutation, "merge_sampling no mutation"),
        (test_extract_text_handler_format, "extract_text handler format"),
        (test_extract_text_handler_format_tokens, "extract_text handler tokens"),
        (test_extract_text_raw_vllm_format, "extract_text raw vLLM format"),
        (test_extract_text_raw_vllm_text, "extract_text raw vLLM text"),
        (test_extract_text_string_output, "extract_text string output"),
        (test_extract_text_root_text, "extract_text root text"),
        (test_extract_text_dict_tokens_objects, "extract_text dict token objects"),
        (test_select_variant_resume_senior_lead, "select_variant resume senior_lead"),
        (test_select_variant_resume_technical_depth, "select_variant resume technical_depth"),
        (test_select_variant_jd_dual_variant, "select_variant JD dual variant"),
        (test_select_variant_jd_tech_only, "select_variant JD tech only"),
        (test_select_variant_jd_remote_only, "select_variant JD remote only"),
        (test_select_variant_recruiting_streaming, "select_variant recruiting streaming"),
        (test_select_variant_recruiting_senior_exec, "select_variant recruiting senior exec"),
        (test_select_variant_match_quick_screen, "select_variant match quick_screen"),
        (test_select_variant_ats_role_specific, "select_variant ATS role_specific"),
        (test_select_variant_no_match, "select_variant no match"),
        (test_extract_slots_role, "extract_slots role"),
        (test_extract_slots_stack_multiple, "extract_slots multiple stacks"),
        (test_extract_slots_go_with_boundaries, "extract_slots go with boundaries"),
        (test_extract_slots_golang, "extract_slots golang"),
        (test_extract_slots_go_and_golang, "extract_slots both go and golang"),
        (test_extract_slots_location_sf, "extract_slots location SF"),
        (test_extract_slots_location_nyc, "extract_slots location NYC"),
        (test_extract_slots_location_remote, "extract_slots location remote"),
        (test_extract_slots_no_false_positives, "extract_slots no false positives"),
        (test_extract_slots_combined, "extract_slots combined"),
        (test_templates_have_seed_flag, "templates have seed flag"),
        (test_templates_job_resume_match_has_sampling, "job_resume_match has sampling"),
    ]

    passed = 0
    failed = 0

    print(f"\n{'='*70}")
    print("Running EvalMatch Career Copilot Unit Tests")
    print(f"{'='*70}\n")

    for test_func, description in test_functions:
        try:
            test_func()
            print(f"✅ PASS: {description}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {description}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"   Error: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
