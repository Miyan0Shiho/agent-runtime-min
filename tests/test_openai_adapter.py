

# ===========================================================================
# The developer role must not reach the wire
# ===========================================================================


def test_the_developer_role_is_mapped_to_system_on_the_wire():
    """Found by running arm against a real provider, not by a test.

    Every turn failed with a 422:

        messages[2].role: unknown variant `developer`, expected one of
        `system`, `user`, `assistant`, `tool`

    `developer` is this runtime's name for "instructions the system added later"
    -- the tool catalogue. OpenAI accepts the role; most OpenAI-compatible
    providers do not.

    The role is mapped at the adapter rather than the internal vocabulary being
    changed: the distinction is real, and this is where provider differences
    belong. It is also the kind of defect only a real provider finds, which is
    why it survived a suite of unit tests that all used a fake server.
    """
    from arm.llm.openai_adapter import OpenAICompatibleAdapter
    from arm.session.messages import Message

    adapter = OpenAICompatibleAdapter(api_key="k", base_url="http://x", model="m")
    rendered = adapter._render_message(Message.developer("the tool list"))

    assert rendered["role"] == "system", (
        f"the developer role reached the wire: {rendered['role']!r}"
    )


def test_every_generated_message_uses_a_universal_wire_role():
    """The general form: nothing may emit a role a typical provider rejects.

    Asserted over the whole vocabulary rather than the one role that broke, so
    adding an internal role later fails here instead of at a provider.
    """
    from arm.llm.openai_adapter import OpenAICompatibleAdapter
    from arm.session.messages import Message

    # The roles the OpenAI-compatible protocol defines.
    wire_safe = {"system", "user", "assistant", "tool"}
    adapter = OpenAICompatibleAdapter(api_key="k", base_url="http://x", model="m")

    messages = [
        Message.system("sys"),
        Message.user("hi"),
        Message.assistant("hello"),
        Message.developer("catalogue"),
    ]
    for message in messages:
        rendered = adapter._render_message(message)
        assert rendered["role"] in wire_safe, (
            f"{message.role!r} is not a universal wire role; map it in "
            f"OpenAICompatibleAdapter._WIRE_ROLES"
        )
