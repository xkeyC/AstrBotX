<template>
  <div class="infra-panel">
    <div class="infra-header">
      <div class="infra-headline">
        <div class="infra-title">{{ tm('localInfra.title') }}</div>
        <div class="infra-subtitle">{{ tm('localInfra.subtitle') }}</div>
      </div>
      <div class="d-flex align-center ga-2 flex-wrap">
        <v-btn
          variant="text"
          rounded="xl"
          prepend-icon="mdi-refresh"
          :loading="refreshing"
          :disabled="!saved.enable"
          @click="refresh"
        >
          {{ tm('localInfra.refresh') }}
        </v-btn>
        <v-btn
          color="primary"
          variant="tonal"
          rounded="xl"
          prepend-icon="mdi-content-save-outline"
          :loading="saving"
          :disabled="!modified"
          @click="save"
        >
          {{ tm('localInfra.save') }}
        </v-btn>
      </div>
    </div>

    <v-divider />

    <div v-if="loading" class="pa-8 d-flex justify-center">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <div v-else-if="form" class="infra-body">
      <section class="infra-section">
        <div class="infra-section-title">{{ tm('localInfra.connection') }}</div>
        <v-switch
          v-model="form.enable"
          :label="tm('localInfra.enable')"
          :hint="tm('localInfra.enableHint')"
          persistent-hint
          color="primary"
          inset
          density="comfortable"
          class="mb-2"
        />
        <div class="infra-grid">
          <v-text-field
            v-model="form.base_url"
            :label="tm('localInfra.baseUrl')"
            :hint="tm('localInfra.baseUrlHint')"
            persistent-hint
            placeholder="http://127.0.0.1:17890"
            variant="outlined"
            density="comfortable"
          />
          <v-text-field
            v-model="form.token"
            :label="tm('localInfra.token')"
            :hint="tm('localInfra.tokenHint')"
            persistent-hint
            :type="showToken ? 'text' : 'password'"
            :append-inner-icon="showToken ? 'mdi-eye-off' : 'mdi-eye'"
            autocomplete="off"
            variant="outlined"
            density="comfortable"
            @click:append-inner="showToken = !showToken"
          />
          <v-text-field
            v-model.number="form.timeout"
            :label="tm('localInfra.timeout')"
            :hint="tm('localInfra.timeoutHint')"
            persistent-hint
            type="number"
            min="5"
            variant="outlined"
            density="comfortable"
          />
          <div class="d-flex align-center">
            <v-btn
              variant="outlined"
              rounded="xl"
              prepend-icon="mdi-connection"
              :loading="probing"
              @click="probe"
            >
              {{ tm('localInfra.probe') }}
            </v-btn>
          </div>
        </div>
        <v-alert
          v-if="statusAlert"
          :type="statusAlert.type"
          variant="tonal"
          density="compact"
          class="mt-4"
        >
          {{ statusAlert.text }}
        </v-alert>
      </section>

      <v-divider />

      <section class="infra-section">
        <div class="infra-section-title">{{ tm('localInfra.services') }}</div>
        <div class="infra-hint mb-3">{{ tm('localInfra.servicesHint') }}</div>

        <div v-for="name in serviceNames" :key="name" class="infra-service">
          <div class="infra-service-row">
            <div class="infra-service-name">
              <v-icon :icon="serviceIcons[name]" size="20" class="mr-2" />
              <div>
                <div class="font-weight-medium">{{ tm(`localInfra.service.${name}.name`) }}</div>
                <div class="infra-hint">{{ tm(`localInfra.service.${name}.hint`) }}</div>
              </div>
            </div>
            <v-switch
              v-model="form.services[name].enable"
              color="primary"
              inset
              hide-details
              density="compact"
              class="infra-service-switch"
            />
            <v-select
              v-if="name !== 'moderation' && name !== 'realtime_voice'"
              v-model="form.services[name].model"
              :items="modelItems(name)"
              :label="tm('localInfra.model')"
              :disabled="!form.services[name].enable"
              hide-details
              variant="outlined"
              density="compact"
              class="infra-service-model"
            />
            <div v-else class="infra-service-model infra-hint">
              {{ name === 'realtime_voice' ? realtimeUrl : tm('localInfra.serverPicks') }}
            </div>
            <v-chip
              :color="statusOf(name).color"
              size="small"
              variant="tonal"
              class="infra-service-status"
            >
              {{ statusOf(name).text }}
            </v-chip>
          </div>

          <div v-if="name === 'tts' && form.services.tts.enable" class="infra-grid infra-service-extra">
            <v-text-field
              v-model="form.services.tts.reference_audio"
              :label="tm('localInfra.refAudio')"
              :hint="tm('localInfra.refAudioHint')"
              persistent-hint
              variant="outlined"
              density="comfortable"
            />
            <v-text-field
              v-model="form.services.tts.reference_text"
              :label="tm('localInfra.refText')"
              :hint="tm('localInfra.refTextHint')"
              persistent-hint
              variant="outlined"
              density="comfortable"
            />
          </div>

          <div
            v-if="name === 'realtime_voice' && form.services.realtime_voice.enable"
            class="infra-grid infra-service-extra"
          >
            <v-select
              v-model="form.services.realtime_voice.asr_model"
              :items="realtimeModelItems('asr_model', 'stt')"
              :label="tm('localInfra.realtimeAsr')"
              :hint="tm('localInfra.realtimeAsrHint')"
              persistent-hint
              variant="outlined"
              density="comfortable"
            />
            <v-select
              v-model="form.services.realtime_voice.tts_model"
              :items="realtimeModelItems('tts_model', 'tts')"
              :label="tm('localInfra.realtimeTts')"
              :hint="tm('localInfra.realtimeTtsHint')"
              persistent-hint
              variant="outlined"
              density="comfortable"
            />
          </div>

          <div v-if="name === 'moderation' && form.services.moderation.enable" class="infra-service-extra">
            <div class="infra-grid">
              <v-text-field
                v-model="form.services.moderation.blocked_reply"
                :label="tm('localInfra.moderation.blockedReply')"
                :hint="tm('localInfra.moderation.blockedReplyHint')"
                persistent-hint
                variant="outlined"
                density="comfortable"
              />
              <div />
              <v-text-field
                v-model.number="form.services.moderation.threshold"
                :label="tm('localInfra.moderation.threshold')"
                :hint="tm('localInfra.moderation.thresholdHint')"
                persistent-hint
                type="number"
                step="0.05"
                min="0"
                max="0.99"
                variant="outlined"
                density="comfortable"
              />
              <v-text-field
                v-model.number="form.services.moderation.nsfw_threshold"
                :label="tm('localInfra.moderation.nsfwThreshold')"
                :hint="tm('localInfra.moderation.nsfwThresholdHint')"
                persistent-hint
                type="number"
                step="0.05"
                min="0"
                max="0.99"
                variant="outlined"
                density="comfortable"
              />
            </div>
            <div class="infra-subtitle-2 mt-4">{{ tm('localInfra.moderation.nsfwLabels') }}</div>
            <div class="infra-hint">{{ tm('localInfra.moderation.nsfwLabelsHint') }}</div>
            <div class="d-flex flex-wrap">
              <v-checkbox
                v-for="label in nsfwLabels"
                :key="label"
                v-model="form.services.moderation.nsfw_labels"
                :value="label"
                :label="tm(`localInfra.moderation.nsfw.${label}`)"
                hide-details
                density="compact"
                class="mr-4"
              />
            </div>
            <div class="infra-subtitle-2 mt-4">{{ tm('localInfra.moderation.categories') }}</div>
            <div class="infra-hint">{{ tm('localInfra.moderation.categoriesHint') }}</div>
            <div class="d-flex flex-wrap">
              <v-checkbox
                v-for="category in categories"
                :key="category"
                v-model="form.services.moderation.categories"
                :value="category"
                :label="tm(`localInfra.moderation.category.${category}`)"
                hide-details
                density="compact"
                class="mr-4"
              />
            </div>
          </div>
        </div>
      </section>

      <v-divider />

      <section class="infra-section">
        <div class="infra-section-title">{{ tm('localInfra.models') }}</div>
        <div class="infra-hint mb-3">{{ tm('localInfra.modelsHint') }}</div>
        <div v-if="!models.length" class="infra-hint">{{ tm('localInfra.noModels') }}</div>
        <v-table v-else density="compact" class="infra-models">
          <thead>
            <tr>
              <th>{{ tm('localInfra.modelId') }}</th>
              <th>{{ tm('localInfra.taskKinds') }}</th>
              <th>{{ tm('localInfra.modelState') }}</th>
              <th>{{ tm('localInfra.usedBy') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="model in models" :key="model.id">
              <td>
                <div class="font-weight-medium">{{ model.id }}</div>
                <div v-if="model.name" class="infra-hint">{{ model.name }}</div>
              </td>
              <td>
                <v-chip
                  v-for="kind in model.task_kinds"
                  :key="kind"
                  size="x-small"
                  variant="outlined"
                  class="mr-1 mb-1"
                >
                  {{ kind }}
                </v-chip>
              </td>
              <td>
                <v-chip :color="modelState(model).color" size="x-small" variant="tonal">
                  {{ modelState(model).text }}
                </v-chip>
              </td>
              <td>
                <span class="infra-hint">{{ usedBy(model.id) }}</span>
              </td>
            </tr>
          </tbody>
        </v-table>
      </section>
    </div>

    <v-snackbar v-model="snackbar.show" :color="snackbar.color" :timeout="3000" location="top">
      {{ snackbar.message }}
    </v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { httpClient } from '@/api/http'
import { useModuleI18n } from '@/i18n/composables'

type ServiceName = 'stt' | 'tts' | 'embedding' | 'rerank' | 'moderation' | 'realtime_voice'

interface InfraModel {
  id: string
  name: string
  task_kinds: string[]
  enabled: boolean
  downloaded: boolean | null
  usable: boolean
}

interface ServiceState {
  service: ServiceName
  task_kind: string
  switched_on: boolean
  available: boolean
  model: string
  chosen: string
  candidates: string[]
  reason: string
  provider_id?: string
}

interface InfraState {
  settings: Record<string, any>
  services: Record<ServiceName, ServiceState>
  realtime_url: string
  catalog: { known: boolean; models: InfraModel[] | null; error: string; fetched_at: number }
}

const { tm } = useModuleI18n('features/provider')

const serviceNames: ServiceName[] = ['stt', 'tts', 'embedding', 'rerank', 'moderation', 'realtime_voice']
const serviceIcons: Record<ServiceName, string> = {
  stt: 'mdi-microphone-message',
  tts: 'mdi-volume-high',
  embedding: 'mdi-code-json',
  rerank: 'mdi-compare-vertical',
  moderation: 'mdi-shield-check-outline',
  realtime_voice: 'mdi-phone-in-talk'
}
const taskKinds: Record<ServiceName, string> = {
  stt: 'asr.transcribe',
  tts: 'tts.synthesize',
  embedding: 'text.embed',
  rerank: 'text.rerank',
  moderation: 'text.moderate',
  realtime_voice: 'voice.realtime'
}
const nsfwLabels = ['low', 'medium', 'high']
const categories = [
  'violent',
  'illegal',
  'sexual',
  'pii',
  'self-harm',
  'unethical',
  'political',
  'copyright',
  'jailbreak'
]

const state = ref<InfraState | null>(null)
const form = ref<Record<string, any> | null>(null)
const probedModels = ref<InfraModel[] | null>(null)
const loading = ref(true)
const saving = ref(false)
const refreshing = ref(false)
const probing = ref(false)
const showToken = ref(false)
const snackbar = ref({ show: false, message: '', color: 'success' })

function notify(message: string, color = 'success') {
  snackbar.value = { show: true, message, color }
}

function errorText(e: any, fallback: string) {
  return e?.response?.data?.message || e?.message || fallback
}

const saved = computed(() => state.value?.settings ?? { enable: false })
const modified = computed(
  () => !!form.value && JSON.stringify(form.value) !== JSON.stringify(state.value?.settings)
)
const models = computed<InfraModel[]>(
  () => probedModels.value ?? state.value?.catalog.models ?? []
)
const realtimeUrl = computed(() => {
  const base = String(form.value?.base_url || '').replace(/\/+$/, '').replace(/\/v1$/, '')
  if (!base) return ''
  return base.replace(/^http/, 'ws') + '/v1/realtime'
})

const statusAlert = computed(() => {
  if (!state.value || !saved.value.enable) {
    return { type: 'info' as const, text: tm('localInfra.statusOff') }
  }
  const catalog = state.value.catalog
  if (catalog.error) {
    return { type: 'error' as const, text: tm('localInfra.statusError', { error: catalog.error }) }
  }
  if (catalog.models && !catalog.known) {
    return { type: 'warning' as const, text: tm('localInfra.statusOld') }
  }
  if (catalog.models) {
    const usable = catalog.models.filter((m) => m.usable).length
    return {
      type: 'success' as const,
      text: tm('localInfra.statusOk', { usable, total: catalog.models.length })
    }
  }
  return null
})

function apply(data: InfraState) {
  state.value = data
  form.value = JSON.parse(JSON.stringify(data.settings))
  probedModels.value = null
}

async function load() {
  loading.value = true
  try {
    const res = await httpClient.get('/api/local_infra')
    if (res.data.status === 'ok') apply(res.data.data)
    else notify(res.data.message || tm('localInfra.loadFailed'), 'error')
  } catch (e) {
    notify(errorText(e, tm('localInfra.loadFailed')), 'error')
  } finally {
    loading.value = false
  }
}

async function save() {
  if (!form.value) return
  saving.value = true
  try {
    const res = await httpClient.post('/api/local_infra', form.value)
    if (res.data.status === 'ok') {
      apply(res.data.data)
      notify(tm('localInfra.saved'))
    } else {
      notify(res.data.message || tm('localInfra.saveFailed'), 'error')
    }
  } catch (e) {
    notify(errorText(e, tm('localInfra.saveFailed')), 'error')
  } finally {
    saving.value = false
  }
}

async function refresh() {
  refreshing.value = true
  try {
    const res = await httpClient.post('/api/local_infra/refresh')
    if (res.data.status === 'ok') {
      const draft = form.value
      const wasModified = modified.value
      apply(res.data.data)
      if (wasModified && draft) form.value = draft
    } else {
      notify(res.data.message || tm('localInfra.loadFailed'), 'error')
    }
  } catch (e) {
    notify(errorText(e, tm('localInfra.loadFailed')), 'error')
  } finally {
    refreshing.value = false
  }
}

async function probe() {
  if (!form.value) return
  probing.value = true
  try {
    const res = await httpClient.post('/api/local_infra/probe', form.value)
    if (res.data.status === 'ok') {
      probedModels.value = res.data.data.models
      notify(tm('localInfra.probeOk', { count: res.data.data.models.length }))
    } else {
      notify(res.data.message || tm('localInfra.probeFailed'), 'error')
    }
  } catch (e) {
    notify(errorText(e, tm('localInfra.probeFailed')), 'error')
  } finally {
    probing.value = false
  }
}

function candidates(name: ServiceName): string[] {
  return models.value
    .filter((m) => m.usable && m.task_kinds.includes(taskKinds[name]))
    .map((m) => m.id)
}

function modelItems(name: ServiceName) {
  const ids = candidates(name)
  const auto = ids[0]
  const items = [
    {
      title: auto ? tm('localInfra.autoModel', { model: auto }) : tm('localInfra.autoModelNone'),
      value: ''
    },
    ...ids.map((id) => ({ title: id, value: id }))
  ]
  const chosen = form.value?.services[name]?.model
  if (chosen && !ids.includes(chosen)) {
    items.push({ title: tm('localInfra.unavailableModel', { model: chosen }), value: chosen })
  }
  return items
}

function realtimeModelItems(key: 'asr_model' | 'tts_model', service: ServiceName) {
  const ids = candidates(service)
  const items = [
    { title: tm('localInfra.realtimeModelDefault'), value: '' },
    ...ids.map((id) => ({ title: id, value: id }))
  ]
  const chosen = form.value?.services.realtime_voice?.[key]
  if (chosen && !ids.includes(chosen)) {
    items.push({ title: tm('localInfra.unavailableModel', { model: chosen }), value: chosen })
  }
  return items
}

function statusOf(name: ServiceName) {
  const service = state.value?.services[name]
  if (!service || !service.switched_on) {
    return { color: 'grey', text: tm('localInfra.state.off') }
  }
  if (!service.available) {
    return {
      color: 'error',
      text: tm(`localInfra.state.${service.reason || 'no_model'}`)
    }
  }
  if (service.reason === 'unknown') {
    return { color: 'warning', text: tm('localInfra.state.unknown') }
  }
  return {
    color: 'success',
    text: service.model
      ? tm('localInfra.state.using', { model: service.model })
      : tm('localInfra.state.on')
  }
}

function modelState(model: InfraModel) {
  if (!model.enabled) return { color: 'grey', text: tm('localInfra.modelDisabled') }
  // Another variant may serve the task; a service pinned to it shows the error.
  if (model.downloaded === false) return { color: 'grey', text: tm('localInfra.modelNotDownloaded') }
  return { color: 'success', text: tm('localInfra.modelReady') }
}

function usedBy(modelId: string) {
  const services = state.value?.services
  if (!services) return ''
  return serviceNames
    .filter((name) => services[name]?.available && services[name]?.model === modelId)
    .map((name) => tm(`localInfra.service.${name}.name`))
    .join('、')
}

onMounted(load)
</script>

<style scoped>
.infra-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
}

.infra-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 24px;
  flex-wrap: wrap;
}

.infra-title {
  font-size: 18px;
  font-weight: 600;
}

.infra-subtitle,
.infra-hint {
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}

.infra-subtitle-2 {
  font-size: 14px;
  font-weight: 500;
}

.infra-body {
  overflow-y: auto;
  min-height: 0;
}

.infra-section {
  padding: 20px 24px;
}

.infra-section-title {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
}

.infra-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.infra-service {
  border: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  border-radius: 12px;
  padding: 12px 16px;
  margin-bottom: 12px;
}

.infra-service-row {
  display: grid;
  grid-template-columns: minmax(0, 2fr) 64px minmax(0, 1.6fr) minmax(0, 1.2fr);
  align-items: center;
  gap: 16px;
}

.infra-service-status {
  justify-self: start;
  max-width: 100%;
}

.infra-service-status :deep(.v-chip__content) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.infra-service-name {
  display: flex;
  align-items: flex-start;
}

.infra-service-model {
  min-width: 0;
  overflow-wrap: anywhere;
}

.infra-service-extra {
  margin-top: 16px;
}

.infra-models td {
  vertical-align: top;
  padding-top: 8px !important;
  padding-bottom: 8px !important;
}

@media (max-width: 960px) {
  .infra-grid,
  .infra-service-row {
    grid-template-columns: 1fr;
  }

  .infra-header,
  .infra-section {
    padding-left: 16px;
    padding-right: 16px;
  }
}
</style>
