<template>
  <v-expansion-panels variant="accordion" class="provider-extras">
    <v-expansion-panel>
      <v-expansion-panel-title>
        <span class="text-body-2">{{ tm('providers.extras') }}</span>
        <span class="extras-summary">{{ summary }}</span>
      </v-expansion-panel-title>
      <v-expansion-panel-text>
        <div class="extras-block">
          <v-select
            v-model="extras.compaction"
            :items="compactionItems"
            :label="tm('providers.compaction')"
            :hint="tm('providers.compactionHint')"
            persistent-hint
            variant="outlined"
            density="comfortable"
            class="compaction-select"
          />
          <v-switch
            v-model="extras.files_api"
            :label="tm('providers.filesApi')"
            :hint="tm('providers.filesApiHint')"
            persistent-hint
            color="primary"
            density="comfortable"
            inset
          />
        </div>

        <div class="extras-block">
          <v-textarea
            v-model="extras.extra_body_json"
            :label="tm('providers.extraBody')"
            :hint="tm('providers.extraBodyHint')"
            persistent-hint
            auto-grow
            rows="2"
            placeholder='{"thinking": {"type": "disabled"}}'
            class="text-mono"
            variant="outlined"
            density="comfortable"
          />
        </div>

        <div class="extras-block">
          <div class="extras-head">
            <div>
              <div class="text-subtitle-2">{{ tm('providers.headers') }}</div>
              <div class="setting-subtitle">{{ tm('providers.headersHint') }}</div>
            </div>
            <v-btn variant="text" color="primary" prepend-icon="mdi-plus" @click="addHeader">
              {{ tm('providers.addHeader') }}
            </v-btn>
          </div>
          <div v-for="(h, idx) in extras.headers" :key="h.__key" class="header-row">
            <v-text-field
              v-model="h.name"
              :label="tm('providers.headerName')"
              variant="outlined"
              density="compact"
              hide-details
            />
            <v-text-field
              v-model="h.value"
              :label="tm('providers.headerValue')"
              variant="outlined"
              density="compact"
              hide-details
            />
            <v-btn
              icon="mdi-delete-outline"
              variant="text"
              color="error"
              density="comfortable"
              @click="extras.headers.splice(idx, 1)"
            />
          </div>
        </div>

        <div class="extras-block">
          <div class="extras-head">
            <div>
              <div class="text-subtitle-2">{{ tm('providers.models') }}</div>
              <div class="setting-subtitle">{{ tm('providers.modelsHint') }}</div>
            </div>
            <v-btn variant="text" color="primary" prepend-icon="mdi-plus" @click="extras.models.push(emptyModelRow())">
              {{ tm('providers.addModel') }}
            </v-btn>
          </div>
          <div v-for="(m, idx) in extras.models" :key="m.__key" class="model-card">
            <div class="model-grid">
              <v-text-field
                v-model="m.slug"
                :label="tm('providers.modelSlug')"
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
              <v-text-field
                v-model="m.context_window"
                :label="tm('providers.contextWindow')"
                type="number"
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
              <v-text-field
                v-model="m.auto_compact_token_limit"
                :label="tm('providers.autoCompact')"
                type="number"
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
              <v-select
                v-model="m.reasoning_efforts"
                :items="REASONING_EFFORTS"
                :label="tm('providers.reasoningEfforts')"
                multiple
                chips
                closable-chips
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
              <v-select
                v-model="m.default_reasoning_effort"
                :items="defaultEffortItems(m.reasoning_efforts)"
                :label="tm('providers.defaultEffort')"
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
              <v-select
                v-model="m.image_input"
                :items="imageInputItems"
                :label="tm('providers.imageInput')"
                variant="outlined"
                density="compact"
                hide-details="auto"
              />
            </div>
            <v-textarea
              v-model="m.metadata_json"
              :label="tm('providers.metadataJson')"
              :hint="tm('providers.metadataJsonHint')"
              persistent-hint
              rows="2"
              auto-grow
              variant="outlined"
              density="compact"
              class="mt-2 metadata-json"
            />
            <div class="model-actions">
              <v-btn
                variant="text"
                color="error"
                size="small"
                prepend-icon="mdi-delete-outline"
                @click="extras.models.splice(idx, 1)"
              >
                {{ tm('providers.removeModel') }}
              </v-btn>
            </div>
          </div>
        </div>
      </v-expansion-panel-text>
    </v-expansion-panel>
  </v-expansion-panels>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useModuleI18n } from '@/i18n/composables'
import {
  COMPACTION_MODES,
  REASONING_EFFORTS,
  emptyModelRow,
  newKey,
  type ProviderExtras
} from './types'

const props = defineProps<{ modelValue: ProviderExtras }>()
// The provider row owns these settings; they are edited in place.
const extras = computed(() => props.modelValue)
const { tm } = useModuleI18n('features/codex')

const compactionItems = computed(() =>
  COMPACTION_MODES.map((mode) => ({ title: tm(`providers.compaction_${mode}`), value: mode }))
)

const summary = computed(() => {
  const parts: string[] = []
  if (extras.value.compaction !== 'auto') parts.push(tm(`providers.compaction_${extras.value.compaction}`))
  if (extras.value.files_api) parts.push(tm('providers.filesApi'))
  if (extras.value.headers.length) parts.push(tm('providers.headerCount', { n: String(extras.value.headers.length) }))
  if (extras.value.models.length) parts.push(tm('providers.modelCount', { n: String(extras.value.models.length) }))
  return parts.join(' · ')
})

const imageInputItems = computed(() => [
  { title: tm('providers.imageInputDefault'), value: null },
  { title: tm('providers.imageInputYes'), value: true },
  { title: tm('providers.imageInputNo'), value: false }
])

function defaultEffortItems(efforts: string[]) {
  return [{ title: tm('model.reasoningDefault'), value: '' }, ...efforts.map((e) => ({ title: e, value: e }))]
}

function addHeader() {
  extras.value.headers.push({ __key: newKey(), name: '', value: '' })
}
</script>

<style scoped>
.provider-extras {
  grid-column: 1 / -1;
}

.extras-summary {
  margin-left: 12px;
  font-size: 12px;
  color: var(--dashboard-muted);
}

.extras-block + .extras-block {
  margin-top: 18px;
}

.extras-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 8px;
}

.compaction-select {
  max-width: 320px;
}

.header-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2fr) auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}

.model-card {
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 10px;
}

.model-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) repeat(2, minmax(0, 1fr)) minmax(0, 2fr) minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
}

.metadata-json :deep(textarea) {
  font-family: var(--dashboard-mono, monospace);
  font-size: 12px;
}

.model-actions {
  display: flex;
  justify-content: flex-end;
}

@media (max-width: 1280px) {
  .model-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .model-grid,
  .header-row {
    grid-template-columns: 1fr;
  }
}
</style>
