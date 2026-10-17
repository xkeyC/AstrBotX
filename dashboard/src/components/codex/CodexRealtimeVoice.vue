<template>
  <div>
    <v-btn-toggle v-model="voice.backend" mandatory color="primary" variant="outlined" density="comfortable" class="mb-4">
      <v-btn value="builtin">{{ tm('voice.backendBuiltin') }}</v-btn>
      <v-btn value="local_infra">{{ tm('voice.backendInfra') }}</v-btn>
    </v-btn-toggle>
    <div class="voice-hint mb-4">
      {{ voice.backend === 'local_infra' ? tm('voice.backendInfraHint') : tm('voice.backendBuiltinHint') }}
    </div>

    <div v-if="voice.backend === 'builtin'" class="voice-grid">
      <v-select
        v-model="voice.voice"
        :items="voiceItems"
        :label="tm('voice.voice')"
        :hint="tm('voice.voiceHint')"
        persistent-hint
        variant="outlined"
        density="comfortable"
      />
      <v-text-field
        v-model="voice.model"
        :label="tm('voice.realtimeModel')"
        :hint="tm('voice.realtimeModelHint')"
        persistent-hint
        variant="outlined"
        density="comfortable"
      />
    </div>

    <template v-else>
      <div class="text-subtitle-2 mb-2">{{ tm('voice.serverTitle') }}</div>
      <div class="voice-hint mb-4">
        {{ tm('voice.infraServerHint') }}
        <router-link to="/providers">{{ tm('voice.infraServerLink') }}</router-link>
      </div>
      <div class="voice-grid mb-4">
        <v-text-field
          v-model="voice.ref_audio"
          :label="tm('voice.refAudio')"
          :hint="tm('voice.refAudioHint')"
          persistent-hint
          variant="outlined"
          density="comfortable"
        />
        <v-switch
          v-model="voice.stream_text"
          :label="tm('voice.streamText')"
          :hint="tm('voice.streamTextHint')"
          persistent-hint
          color="primary"
          inset
          density="comfortable"
        />
        <v-textarea
          v-model="voice.ref_text"
          class="voice-wide"
          :label="tm('voice.refText')"
          :hint="tm('voice.refTextHint')"
          :disabled="!voice.ref_audio.trim() && !voice.ref_text"
          persistent-hint
          auto-grow
          rows="2"
          variant="outlined"
          density="comfortable"
        />
        <v-select
          v-model="voice.emotion"
          :items="emotionItems"
          :label="tm('voice.emotion')"
          :hint="tm('voice.emotionHint')"
          persistent-hint
          variant="outlined"
          density="comfortable"
        />
        <v-text-field
          v-model.number="voice.emotion_strength"
          :label="tm('voice.emotionStrength')"
          :hint="tm('voice.emotionStrengthHint')"
          persistent-hint
          type="number"
          step="0.1"
          min="0"
          max="1"
          variant="outlined"
          density="comfortable"
        />
      </div>

      <div class="text-subtitle-2 mb-2">{{ tm('voice.textTitle') }}</div>
      <div class="voice-grid">
        <v-select
          v-model="voice.text_model_provider"
          :items="textProviderItems"
          :label="tm('voice.textProvider')"
          :hint="tm('voice.textProviderHint')"
          persistent-hint
          variant="outlined"
          density="comfortable"
        />
        <v-combobox
          v-model="voice.text_model"
          :items="modelIds"
          :label="tm('voice.textModel')"
          :hint="tm('voice.textModelHint')"
          persistent-hint
          clearable
          variant="outlined"
          density="comfortable"
        />
        <v-select
          v-model="voice.text_reasoning_effort"
          :items="effortItems"
          :label="tm('voice.textEffort')"
          :hint="tm('voice.textEffortHint')"
          persistent-hint
          variant="outlined"
          density="comfortable"
        />
        <v-text-field
          v-model.number="voice.idle_compact_percent"
          :label="tm('voice.idleCompact')"
          :hint="tm('voice.idleCompactHint')"
          persistent-hint
          type="number"
          min="0"
          max="100"
          suffix="%"
          variant="outlined"
          density="comfortable"
        />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useModuleI18n } from '@/i18n/composables'
import { EMOTIONS, REALTIME_VOICES, REASONING_EFFORTS, type RealtimeVoiceForm } from './types'

const props = defineProps<{
  modelValue: RealtimeVoiceForm
  // The model providers (built-in and custom), as the page lists them.
  providerItems: { title: string; value: string }[]
  modelIds: string[]
}>()
// The page owns these settings; they are edited in place.
const voice = computed(() => props.modelValue)
const { tm } = useModuleI18n('features/codex')

const voiceItems = computed(() => [
  { title: tm('voice.voiceDefault'), value: '' },
  ...REALTIME_VOICES.map((v) => ({ title: v, value: v }))
])
// The providers, and the chosen one if it was removed meanwhile.
const textProviderItems = computed(() => {
  const current = voice.value.text_model_provider
  if (!current || props.providerItems.some((item) => item.value === current)) return props.providerItems
  return [...props.providerItems, { title: tm('model.providerMissing', { id: current }), value: current }]
})
const emotionItems = computed(() => EMOTIONS.map((e) => ({ title: tm(`voice.emotion_${e}`), value: e })))
const effortItems = computed(() => [
  { title: tm('model.reasoningDefault'), value: '' },
  ...REASONING_EFFORTS.map((e) => ({ title: e, value: e }))
])
</script>

<style scoped>
/* The page's shared form styles are scoped to the page: this section has
   its own. */
.voice-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.voice-wide {
  grid-column: 1 / -1;
}

.voice-hint {
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}

@media (max-width: 1080px) {
  .voice-grid {
    grid-template-columns: 1fr;
  }
}
</style>
