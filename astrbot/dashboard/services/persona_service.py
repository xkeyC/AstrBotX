from __future__ import annotations

from astrbot.core.core_lifecycle import AstrBotCoreLifecycle
from astrbot.core.sentinels import NOT_GIVEN


class PersonaServiceError(Exception):
    pass


class PersonaService:
    def __init__(self, core_lifecycle: AstrBotCoreLifecycle) -> None:
        self.persona_mgr = core_lifecycle.persona_mgr

    async def list_personas(
        self,
        folder_id: str | None,
        filter_by_folder: bool,
    ) -> list[dict]:
        if filter_by_folder:
            personas = await self.persona_mgr.get_personas_by_folder(
                folder_id if folder_id else None
            )
        else:
            personas = await self.persona_mgr.get_all_personas()
        return [self.serialize_persona(persona) for persona in personas]

    async def get_persona_detail(self, data: object) -> dict:
        payload = self._payload(data)
        persona_id = payload.get("persona_id")

        if not persona_id:
            raise PersonaServiceError("缺少必要参数: persona_id")

        persona = await self.persona_mgr.get_persona(persona_id)
        if not persona:
            raise PersonaServiceError("人格不存在")

        return self.serialize_persona(persona)

    async def create_persona(self, data: object) -> dict:
        payload = self._payload(data)
        raw_persona_id = payload.get("persona_id")
        raw_system_prompt = payload.get("system_prompt")
        persona_id = str(raw_persona_id).strip() if raw_persona_id is not None else ""
        system_prompt = (
            str(raw_system_prompt).strip() if raw_system_prompt is not None else ""
        )
        begin_dialogs = payload.get("begin_dialogs", [])
        tools = payload.get("tools")
        skills = payload.get("skills")
        custom_error_message = self._normalize_optional_text(
            payload.get("custom_error_message"), "自定义报错回复信息必须是字符串"
        )
        voice_prompt = self._normalize_optional_text(
            payload.get("voice_prompt"), "语音人格必须是字符串"
        )
        folder_id = payload.get("folder_id")
        sort_order = payload.get("sort_order", 0)

        if not persona_id:
            raise PersonaServiceError("人格ID不能为空")
        if not system_prompt:
            raise PersonaServiceError("系统提示词不能为空")

        self._validate_begin_dialogs(begin_dialogs)

        persona = await self.persona_mgr.create_persona(
            persona_id=persona_id,
            system_prompt=system_prompt,
            begin_dialogs=begin_dialogs if begin_dialogs else None,
            tools=tools,
            skills=skills,
            custom_error_message=custom_error_message,
            voice_prompt=voice_prompt,
            folder_id=folder_id,
            sort_order=sort_order,
        )

        return {
            "message": "人格创建成功",
            "persona": self.serialize_persona(persona, empty_lists_for_tools=True),
        }

    async def update_persona(self, data: object) -> dict:
        payload = self._payload(data)
        persona_id = payload.get("persona_id")
        system_prompt = payload.get("system_prompt")
        begin_dialogs = payload.get("begin_dialogs")
        has_tools = "tools" in payload
        tools = payload.get("tools")
        has_skills = "skills" in payload
        skills = payload.get("skills")
        has_custom_error_message = "custom_error_message" in payload
        custom_error_message = payload.get("custom_error_message")
        has_voice_prompt = "voice_prompt" in payload
        voice_prompt = payload.get("voice_prompt")

        if not persona_id:
            raise PersonaServiceError("缺少必要参数: persona_id")

        if has_custom_error_message:
            custom_error_message = self._normalize_optional_text(
                custom_error_message, "自定义报错回复信息必须是字符串"
            )
        if has_voice_prompt:
            voice_prompt = self._normalize_optional_text(
                voice_prompt, "语音人格必须是字符串"
            )

        if begin_dialogs is not None:
            self._validate_begin_dialogs(begin_dialogs)

        update_kwargs = {
            "persona_id": persona_id,
            "system_prompt": system_prompt,
            "begin_dialogs": begin_dialogs,
        }
        if has_tools:
            update_kwargs["tools"] = tools
        if has_skills:
            update_kwargs["skills"] = skills
        if has_custom_error_message:
            update_kwargs["custom_error_message"] = custom_error_message
        if has_voice_prompt:
            update_kwargs["voice_prompt"] = voice_prompt

        await self.persona_mgr.update_persona(**update_kwargs)
        return {"message": "人格更新成功"}

    async def clone_persona(self, data: object) -> dict:
        """Copies a persona under a new id (WebUI persona cloning).

        Args:
            data: ``source_persona_id`` and ``new_persona_id``.

        Returns:
            A message and the new persona.

        Raises:
            PersonaServiceError: A parameter is missing.
        """
        payload = self._payload(data)
        source_persona_id = str(payload.get("source_persona_id") or "").strip()
        new_persona_id = str(payload.get("new_persona_id") or "").strip()
        if not source_persona_id:
            raise PersonaServiceError("缺少必要参数: source_persona_id")
        if not new_persona_id:
            raise PersonaServiceError("新人格ID不能为空")
        persona = await self.persona_mgr.clone_persona(
            source_persona_id=source_persona_id,
            new_persona_id=new_persona_id,
        )
        return {
            "message": "人格克隆成功",
            "persona": self.serialize_persona(persona, empty_lists_for_tools=True),
        }

    async def delete_persona(self, data: object) -> dict:
        payload = self._payload(data)
        persona_id = payload.get("persona_id")

        if not persona_id:
            raise PersonaServiceError("缺少必要参数: persona_id")

        await self.persona_mgr.delete_persona(persona_id)
        return {"message": "人格删除成功"}

    async def move_persona(self, data: object) -> dict:
        payload = self._payload(data)
        persona_id = payload.get("persona_id")
        folder_id = payload.get("folder_id")

        if not persona_id:
            raise PersonaServiceError("缺少必要参数: persona_id")

        await self.persona_mgr.move_persona_to_folder(persona_id, folder_id)
        return {"message": "人格移动成功"}

    async def list_folders(self, parent_id: str | None) -> list[dict]:
        if parent_id == "":
            parent_id = None
        folders = await self.persona_mgr.get_folders(parent_id)
        return [self.serialize_folder(folder) for folder in folders]

    async def get_folder_tree(self):
        return await self.persona_mgr.get_folder_tree()

    async def get_folder_detail(self, data: object) -> dict:
        payload = self._payload(data)
        folder_id = payload.get("folder_id")

        if not folder_id:
            raise PersonaServiceError("缺少必要参数: folder_id")

        folder = await self.persona_mgr.get_folder(folder_id)
        if not folder:
            raise PersonaServiceError("文件夹不存在")

        return self.serialize_folder(folder)

    async def create_folder(self, data: object) -> dict:
        payload = self._payload(data)
        name = str(payload.get("name", "")).strip()
        parent_id = payload.get("parent_id")
        description = payload.get("description")
        sort_order = payload.get("sort_order", 0)

        if not name:
            raise PersonaServiceError("文件夹名称不能为空")

        folder = await self.persona_mgr.create_folder(
            name=name,
            parent_id=parent_id,
            description=description,
            sort_order=sort_order,
        )

        return {
            "message": "文件夹创建成功",
            "folder": self.serialize_folder(folder),
        }

    async def update_folder(self, data: object) -> dict:
        payload = self._payload(data)
        folder_id = payload.get("folder_id")
        name = payload.get("name")
        parent_id = payload.get("parent_id") if "parent_id" in payload else NOT_GIVEN
        description = (
            payload.get("description") if "description" in payload else NOT_GIVEN
        )
        sort_order = payload.get("sort_order")

        if not folder_id:
            raise PersonaServiceError("缺少必要参数: folder_id")

        await self.persona_mgr.update_folder(
            folder_id=folder_id,
            name=name,
            parent_id=parent_id,
            description=description,
            sort_order=sort_order,
        )

        return {"message": "文件夹更新成功"}

    async def delete_folder(self, data: object) -> dict:
        payload = self._payload(data)
        folder_id = payload.get("folder_id")

        if not folder_id:
            raise PersonaServiceError("缺少必要参数: folder_id")

        await self.persona_mgr.delete_folder(folder_id)
        return {"message": "文件夹删除成功"}

    async def reorder_items(self, data: object) -> dict:
        payload = self._payload(data)
        items = payload.get("items", [])

        if not items:
            raise PersonaServiceError("items 不能为空")

        for item in items:
            if not all(key in item for key in ("id", "type", "sort_order")):
                raise PersonaServiceError(
                    "每个 item 必须包含 id, type, sort_order 字段"
                )
            if item["type"] not in ("persona", "folder"):
                raise PersonaServiceError("type 字段必须是 'persona' 或 'folder'")

        await self.persona_mgr.batch_update_sort_order(items)
        return {"message": "排序更新成功"}

    @staticmethod
    def serialize_persona(persona, empty_lists_for_tools: bool = False) -> dict:
        return {
            "persona_id": persona.persona_id,
            "system_prompt": persona.system_prompt,
            "begin_dialogs": persona.begin_dialogs or [],
            "tools": (persona.tools or []) if empty_lists_for_tools else persona.tools,
            "skills": (persona.skills or [])
            if empty_lists_for_tools
            else persona.skills,
            "custom_error_message": persona.custom_error_message,
            "voice_prompt": persona.voice_prompt,
            "folder_id": persona.folder_id,
            "sort_order": persona.sort_order,
            "created_at": persona.created_at.isoformat()
            if persona.created_at
            else None,
            "updated_at": persona.updated_at.isoformat()
            if persona.updated_at
            else None,
        }

    @staticmethod
    def serialize_folder(folder) -> dict:
        return {
            "folder_id": folder.folder_id,
            "name": folder.name,
            "parent_id": folder.parent_id,
            "description": folder.description,
            "sort_order": folder.sort_order,
            "created_at": folder.created_at.isoformat() if folder.created_at else None,
            "updated_at": folder.updated_at.isoformat() if folder.updated_at else None,
        }

    @staticmethod
    def _normalize_optional_text(value, type_error: str):
        """Strip an optional text field, turning blank strings into None.

        Args:
            value: Raw field value from the request payload.
            type_error: Error message raised when the value is not a string.

        Returns:
            The stripped string, or None when the value is missing or blank.

        Raises:
            PersonaServiceError: If the value is neither None nor a string.
        """
        if value is not None:
            if not isinstance(value, str):
                raise PersonaServiceError(type_error)
            return value.strip() or None
        return None

    @staticmethod
    def _validate_begin_dialogs(begin_dialogs) -> None:
        if begin_dialogs and len(begin_dialogs) % 2 != 0:
            raise PersonaServiceError("预设对话数量必须为偶数（用户和助手轮流对话）")

    @staticmethod
    def _payload(data: object) -> dict:
        return data if isinstance(data, dict) else {}
