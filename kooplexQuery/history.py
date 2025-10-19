from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
from typing import Iterator, List, Dict, Any

class MessageWithMeta:
    def __init__(self, message: BaseMessage, metadata: Dict):
        self.message = message
        self.metadata = metadata

class CustomChatHistory(BaseChatMessageHistory):
    def __init__(self, context: str = None):
        self.context=context
        self.messages_with_meta: List[MessageWithMeta] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return [m.message for m in self.messages_with_meta]

    def add_user_message(self, content: str, metadata: Dict = None):
        if self.is_empty and self.context:
            content="Context:\n"+self.context+"\n\nQuestion:\n"+content
        self.messages_with_meta.append(
            MessageWithMeta(HumanMessage(content=content), metadata or {})
        )

    def add_ai_message(self, content: str, metadata: Dict = None):
        self.messages_with_meta.append(
            MessageWithMeta(AIMessage(content=content), metadata or {})
        )

    def add_system_message(self, content: str, metadata: Dict = None):
        self.messages_with_meta.append(
            MessageWithMeta(SystemMessage(content=content), metadata or {})
        )

    def __len__(self) -> int:
        return len(self.messages_with_meta)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        i = 1
        l = len(self.messages_with_meta)
        while i < l - 1:
            first = self.messages_with_meta[i]
            second = self.messages_with_meta[i + 1]
            if first.message.type == "human" and second.message.type == "ai":
                is_last_pair = (i >= l - 3)
                if i == 1 and first.message.content.startswith("Context:\n"):
                    try:
                        # Split off context and question
                        _, context_part = first.message.content.split("Context:\n", 1)
                        context_str, question = context_part.split("\n\nQuestion:\n", 1)
                    except ValueError:
                        # fallback if split fails
                        context_str = self.context
                        question = first.message.content
                    yield {
                        #"context": context_str.strip(),
                        "question": question.strip(),
                        "question_meta": first.metadata,
                        "answer": second.message.content.strip(),
                        "answer_meta": second.metadata,
                        "is_last": is_last_pair
                    }
                else:
                    yield {
                        "question": first.message.content.strip(),
                        "question_meta": first.metadata,
                        "answer": second.message.content.strip(),
                        "answer_meta": second.metadata,
                        "is_last": is_last_pair
                    }
                i += 2
            else:
                i += 1  # skip if not a valid user/ai pair


    @property
    def is_empty(self) -> bool:
        return len(self)<2

    def clear(self):
        self.messages_with_meta.clear()

    def pop(self):
        if not self.messages_with_meta:
            return
        if self.messages_with_meta[-1].message.type == "ai":
            self.messages_with_meta.pop()
        if self.messages_with_meta and self.messages_with_meta[-1].message.type == "human":
            self.messages_with_meta.pop()

    def filter(self, types: List[str] = None ) -> Iterator[Dict[str, Any]]:
        for r in self:
            if r['question_meta'].get('type') in types:
                continue
            yield r
