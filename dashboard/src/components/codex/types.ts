// Form shapes of the Codex page's realtime voice settings and of the extra
// settings of a custom provider, with their conversion from and to the
// runner config (see astrbot/core/config/agent_runner.py).

export type HeaderRow = { __key: string; name: string; value: string }

export type ModelRow = {
  __key: string
  slug: string
  context_window: string
  auto_compact_token_limit: string
  reasoning_efforts: string[]
  default_reasoning_effort: string
  // null: Codex's default (text and images).
  image_input: boolean | null
  metadata_json: string
}

export type ProviderExtras = {
  headers: HeaderRow[]
  compaction: string
  // Chat wire: merged over each request body (a JSON object, as text).
  extra_body_json: string
  // Images uploaded once to the provider's Files API.
  files_api: boolean
  models: ModelRow[]
}

export type RealtimeVoiceForm = {
  backend: string
  voice: string
  model: string
  infra_url: string
  infra_token: string
  ref_audio: string
  ref_text: string
  emotion: string
  emotion_strength: number
  stream_text: boolean
  text_model_provider: string
  text_model: string
  text_reasoning_effort: string
  idle_compact_percent: number
}

// Voices of Codex realtime (v1/v3).
export const REALTIME_VOICES = [
  'juniper',
  'maple',
  'spruce',
  'ember',
  'vale',
  'breeze',
  'arbor',
  'sol',
  'cove'
]
// Emotions of the local server's TTS (IndexTTS-2.5); none keeps the
// reference voice's own.
export const EMOTIONS = [
  'calm',
  'happy',
  'angry',
  'sad',
  'afraid',
  'disgusted',
  'melancholic',
  'surprised',
  'none'
]
export const REASONING_EFFORTS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh']
export const COMPACTION_MODES = ['auto', 'local', 'remote']

let keySeq = 0
export function newKey(): string {
  keySeq += 1
  return `${Date.now().toString(36)}-${keySeq}`
}

function str(v: unknown): string {
  return v === null || v === undefined ? '' : String(v)
}

function num(v: unknown, fallback: number): number {
  if (v === null || v === undefined || String(v).trim() === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}

function blank(v: unknown): boolean {
  return v === null || v === undefined || String(v).trim() === ''
}

export function emptyModelRow(): ModelRow {
  return {
    __key: newKey(),
    slug: '',
    context_window: '',
    auto_compact_token_limit: '',
    reasoning_efforts: [],
    default_reasoning_effort: '',
    image_input: null,
    metadata_json: ''
  }
}

export function extrasFromConfig(p: any): ProviderExtras {
  const headers = p?.headers && typeof p.headers === 'object' ? p.headers : {}
  const models = Array.isArray(p?.models) ? p.models : []
  return {
    headers: Object.entries(headers).map(([name, value]) => ({
      __key: newKey(),
      name,
      value: str(value)
    })),
    compaction: COMPACTION_MODES.includes(str(p?.compaction)) ? str(p.compaction) : 'auto',
    extra_body_json:
      p?.extra_body && typeof p.extra_body === 'object' && Object.keys(p.extra_body).length
        ? JSON.stringify(p.extra_body, null, 2)
        : '',
    files_api: p?.files_api === true,
    models: models
      .filter((m: any) => m && typeof m === 'object')
      .map((m: any) => ({
        __key: newKey(),
        slug: str(m.slug),
        context_window: str(m.context_window || ''),
        auto_compact_token_limit: str(m.auto_compact_token_limit || ''),
        reasoning_efforts: Array.isArray(m.reasoning_efforts) ? m.reasoning_efforts.map(str) : [],
        default_reasoning_effort: str(m.default_reasoning_effort),
        image_input: typeof m.image_input === 'boolean' ? m.image_input : null,
        metadata_json: str(m.metadata_json)
      }))
  }
}

export function extrasPayload(e: ProviderExtras) {
  const headers: Record<string, string> = {}
  for (const h of e.headers) {
    if (h.name.trim()) headers[h.name.trim()] = h.value
  }
  return {
    headers,
    compaction: e.compaction || 'auto',
    // Text still being typed stays text (saving checks it, extrasProblem).
    extra_body: parseExtraBody(e.extra_body_json),
    files_api: e.files_api,
    models: e.models.map((m) => ({
      slug: m.slug.trim(),
      context_window: Number(m.context_window) || 0,
      auto_compact_token_limit: Number(m.auto_compact_token_limit) || 0,
      reasoning_efforts: [...m.reasoning_efforts],
      // A default the model no longer lists goes.
      default_reasoning_effort:
        !m.reasoning_efforts.length || m.reasoning_efforts.includes(m.default_reasoning_effort)
          ? m.default_reasoning_effort || ''
          : '',
      image_input: m.image_input,
      metadata_json: m.metadata_json.trim()
    }))
  }
}

export function realtimeVoiceFromConfig(v: any): RealtimeVoiceForm {
  return {
    backend: str(v?.backend) === 'local_infra' ? 'local_infra' : 'builtin',
    voice: str(v?.voice),
    model: str(v?.model),
    infra_url: str(v?.infra_url) || 'ws://127.0.0.1:17890/v1/realtime',
    infra_token: str(v?.infra_token),
    ref_audio: str(v?.ref_audio),
    ref_text: str(v?.ref_text),
    emotion: str(v?.emotion) || 'calm',
    emotion_strength: num(v?.emotion_strength, 0.8),
    stream_text: v?.stream_text !== false,
    text_model_provider: str(v?.text_model_provider),
    text_model: str(v?.text_model),
    text_reasoning_effort: str(v?.text_reasoning_effort),
    idle_compact_percent: num(v?.idle_compact_percent, 70)
  }
}

export function realtimeVoicePayload(v: RealtimeVoiceForm) {
  return {
    backend: v.backend,
    voice: v.voice || '',
    model: (v.model || '').trim(),
    infra_url: (v.infra_url || '').trim(),
    infra_token: (v.infra_token || '').trim(),
    ref_audio: (v.ref_audio || '').trim(),
    ref_text: (v.ref_text || '').trim(),
    emotion: v.emotion || 'calm',
    emotion_strength: num(v.emotion_strength, 0.8),
    stream_text: v.stream_text !== false,
    text_model_provider: v.text_model_provider || '',
    text_model: (v.text_model || '').trim(),
    text_reasoning_effort: v.text_reasoning_effort || '',
    idle_compact_percent: Math.round(num(v.idle_compact_percent, 70))
  }
}

/** A problem of a provider's extra settings (an i18n key and its params), or null. */
function parseExtraBody(text: string): unknown {
  if (!text.trim()) return {}
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

export function extrasProblem(e: ProviderExtras): [string, Record<string, string>] | null {
  if (e.extra_body_json.trim()) {
    try {
      const parsed = JSON.parse(e.extra_body_json)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return ['messages.extraBodyInvalid', {}]
      }
    } catch {
      return ['messages.extraBodyInvalid', {}]
    }
  }
  const slugs = new Set<string>()
  for (const m of e.models) {
    const slug = m.slug.trim()
    if (!slug) return ['messages.modelSlugRequired', {}]
    if (slugs.has(slug)) return ['messages.modelSlugDuplicate', { slug }]
    slugs.add(slug)
    if (m.metadata_json.trim()) {
      try {
        const parsed = JSON.parse(m.metadata_json)
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          return ['messages.modelMetadataObject', { slug }]
        }
      } catch {
        return ['messages.modelMetadataInvalid', { slug }]
      }
    }
  }
  return null
}

/** A problem of the realtime voice settings (an i18n key and its params), or null. */
export function realtimeVoiceProblem(v: RealtimeVoiceForm): [string, Record<string, string>] | null {
  if (v.backend !== 'local_infra') return null
  if (!/^wss?:\/\/[^/\s]+/.test((v.infra_url || '').trim())) return ['messages.voiceUrlInvalid', {}]
  const strength = Number(v.emotion_strength)
  if (blank(v.emotion_strength) || !(strength >= 0 && strength <= 1)) {
    return ['messages.voiceStrengthInvalid', {}]
  }
  const percent = Number(v.idle_compact_percent)
  if (blank(v.idle_compact_percent) || !(percent >= 0 && percent <= 100)) {
    return ['messages.voiceCompactInvalid', {}]
  }
  return null
}
