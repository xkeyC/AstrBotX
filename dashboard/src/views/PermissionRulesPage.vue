<template>
  <div class="dashboard-page permission-rules-page" :class="{ 'is-dark': isDark }">
    <v-container fluid class="dashboard-shell pa-4 pa-md-6">
      <div class="dashboard-header">
        <div class="dashboard-header-main">
          <h1 class="dashboard-title">{{ tm('page.title') }}</h1>
          <p class="dashboard-subtitle">{{ tm('page.subtitle') }}</p>
        </div>

        <div class="dashboard-header-actions">
          <v-btn variant="text" color="primary" prepend-icon="mdi-refresh" :loading="loading" @click="reload">
            {{ tm('actions.refresh') }}
          </v-btn>
          <v-btn variant="tonal" color="primary" prepend-icon="mdi-content-save" :loading="saving" @click="save">
            {{ tm('actions.save') }}
          </v-btn>
        </div>
      </div>

      <div v-if="hasUnsavedChanges" class="unsaved-banner">
        <v-icon size="18" color="warning">mdi-alert-circle-outline</v-icon>
        <span>{{ tm('messages.unsavedChangesNotice') }}</span>
      </div>

      <!-- Rules -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('rules.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('rules.subtitle') }}</div>
        </div>
        <div class="dashboard-section-actions">
          <v-btn color="primary" variant="tonal" prepend-icon="mdi-plus" @click="addRule">
            {{ tm('rules.add') }}
          </v-btn>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5">
        <div class="syntax-help">
          <v-icon size="18" color="primary">mdi-information-outline</v-icon>
          <div>
            <div>{{ tm('match.syntaxTitle') }}</div>
            <ul class="syntax-list">
              <li><code>&lt;group_id&gt;/&lt;sender_id&gt;</code> — {{ tm('match.syntaxGroupSender') }}</li>
              <li><code>p_&lt;sender_id&gt;</code> — {{ tm('match.syntaxSender') }}</li>
              <li><code>g_&lt;group_id&gt;</code> — {{ tm('match.syntaxGroup') }}</li>
              <li><code>role:admin</code> / <code>role:member</code> — {{ tm('match.syntaxRole') }}</li>
              <li><code>*</code> — {{ tm('match.syntaxAll') }}</li>
            </ul>
          </div>
        </div>

        <div v-if="rules.length === 0" class="dashboard-empty mt-4">
          {{ tm('rules.empty') }}
        </div>

        <div v-else class="rule-list mt-4">
          <div
            v-for="(rule, idx) in rules"
            :key="rule.__key"
            class="rule-card"
            :class="{ 'rule-card--disabled': !rule.enabled, 'rule-card--hit': testResult?.index === idx }"
          >
            <div class="rule-head">
              <span class="rule-index">#{{ idx + 1 }}</span>
              <v-text-field
                v-model="rule.name"
                :label="tm('rules.name')"
                :placeholder="tm('rules.namePlaceholder')"
                variant="outlined"
                density="compact"
                hide-details
                class="rule-name"
              />
              <v-switch
                v-model="rule.enabled"
                :label="rule.enabled ? tm('rules.enabled') : tm('rules.disabled')"
                color="primary"
                density="compact"
                hide-details
                inset
                class="rule-switch"
              />
              <div class="rule-actions">
                <v-btn
                  icon="mdi-arrow-up"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :disabled="idx === 0"
                  :title="tm('rules.moveUp')"
                  @click="moveRule(idx, -1)"
                />
                <v-btn
                  icon="mdi-arrow-down"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :disabled="idx === rules.length - 1"
                  :title="tm('rules.moveDown')"
                  @click="moveRule(idx, 1)"
                />
                <v-btn
                  icon="mdi-content-copy"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :title="tm('rules.duplicate')"
                  @click="duplicateRule(idx)"
                />
                <v-btn
                  icon="mdi-delete-outline"
                  size="small"
                  variant="text"
                  color="error"
                  density="comfortable"
                  :title="tm('rules.delete')"
                  @click="deleteRule(idx)"
                />
                <v-btn
                  :icon="rule.__expanded ? 'mdi-chevron-up' : 'mdi-chevron-down'"
                  size="small"
                  variant="text"
                  density="comfortable"
                  :title="rule.__expanded ? tm('rules.collapse') : tm('rules.expand')"
                  @click="rule.__expanded = !rule.__expanded"
                />
              </div>
            </div>

            <div v-if="!rule.__expanded" class="rule-summary">
              <v-chip
                v-for="cond in rule.match"
                :key="cond"
                size="small"
                label
                :color="conditionError(cond) ? 'error' : 'primary'"
                variant="tonal"
              >
                {{ cond }}
              </v-chip>
              <span v-if="rule.match.length === 0" class="setting-subtitle mt-0">
                {{ isParent(rule) ? tm('match.template') : tm('match.none') }}
              </span>
              <span v-if="ruleSummary(rule)" class="setting-subtitle mt-0">· {{ ruleSummary(rule) }}</span>
            </div>

            <div v-else class="rule-body">
              <v-combobox
                :model-value="rule.match"
                :label="tm('match.label')"
                :hint="tm('match.hint')"
                :error-messages="matchErrors(rule)"
                :delimiters="[',', ' ']"
                persistent-hint
                multiple
                chips
                closable-chips
                variant="outlined"
                density="comfortable"
                @update:model-value="(v: unknown) => (rule.match = cleanList(v))"
              >
                <template #chip="{ props: chipProps, item }">
                  <v-chip
                    v-bind="chipProps"
                    :color="conditionError(String(item.raw)) ? 'error' : 'primary'"
                    size="small"
                    label
                  >
                    {{ item.raw }}
                  </v-chip>
                </template>
              </v-combobox>

              <v-select
                v-model="rule.inherits"
                :items="parentItems(rule)"
                :label="tm('fields.inherits')"
                :hint="tm('fields.inheritsHint')"
                persistent-hint
                variant="outlined"
                density="comfortable"
                class="mt-4"
              />

              <div class="section-label">{{ tm('fields.toolsTitle') }}</div>
              <div class="dashboard-form-grid">
                <v-combobox
                  v-for="field in patternFields"
                  :key="field.key"
                  :model-value="rule[field.key]"
                  :items="field.kind === 'tool' ? toolNames : mcpServerNames"
                  :label="tm(field.label)"
                  :hint="tm(field.hint)"
                  :loading="field.kind === 'tool' ? toolsLoading : mcpLoading"
                  :delimiters="[',', ' ']"
                  persistent-hint
                  multiple
                  chips
                  closable-chips
                  variant="outlined"
                  density="comfortable"
                  @update:model-value="(v: unknown) => (rule[field.key] = cleanList(v))"
                >
                  <template #chip="{ props: chipProps, item }">
                    <v-chip
                      v-bind="chipProps"
                      :color="field.key.endsWith('_deny') ? 'error' : 'success'"
                      size="small"
                      label
                    >
                      {{ item.raw }}
                    </v-chip>
                  </template>
                </v-combobox>
              </div>

              <div class="section-label">{{ tm('fields.overridesTitle') }}</div>
              <div class="dashboard-form-grid">
                <v-select
                  v-model="rule.persona_id"
                  :items="personaItems(rule.persona_id)"
                  :label="tm('fields.persona')"
                  :hint="tm('fields.personaHint')"
                  :loading="personasLoading"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
                <v-combobox
                  :model-value="rule.model"
                  :items="modelIds"
                  :label="tm('fields.model')"
                  :hint="rule.model ? tm('fields.modelCacheWarning') : tm('fields.modelHint')"
                  :loading="modelsLoading"
                  :color="rule.model ? 'warning' : undefined"
                  persistent-hint
                  clearable
                  variant="outlined"
                  density="comfortable"
                  @update:model-value="(v: unknown) => (rule.model = str(v).trim())"
                >
                  <template #item="{ props: itemProps, item }">
                    <v-list-item
                      v-bind="itemProps"
                      :title="modelDisplayName(String(item.raw))"
                      :subtitle="String(item.raw)"
                    />
                  </template>
                </v-combobox>
                <v-select
                  v-model="rule.global_memory"
                  :items="triStateItems"
                  :label="tm('fields.globalMemory')"
                  :hint="tm('fields.globalMemoryHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
              </div>
              <div v-if="rule.model" class="model-warning">
                <v-icon size="16" color="warning">mdi-alert-outline</v-icon>
                <span>{{ tm('fields.modelCacheWarning') }}</span>
              </div>

              <div class="section-label">{{ tm('rateLimit.title') }}</div>
              <div class="dashboard-form-grid">
                <v-select
                  v-model="rule.rate_mode"
                  :items="rateModeItems"
                  :label="tm('rateLimit.mode')"
                  :hint="tm('rateLimit.modeHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
                <v-text-field
                  v-model="rule.rate_limit_reply"
                  :label="tm('rateLimit.reply')"
                  :placeholder="tm('rateLimit.replyPlaceholder')"
                  :hint="tm('rateLimit.replyHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
              </div>
              <div v-if="rule.rate_mode === 'limit'" class="rate-list mt-3">
                <div v-for="(limit, li) in rule.rate_limits" :key="li" class="rate-row">
                  <v-text-field
                    v-model.number="limit.count"
                    type="number"
                    min="1"
                    :label="tm('rateLimit.count')"
                    :error="!rateRowValid(limit)"
                    variant="outlined"
                    density="compact"
                    hide-details
                  />
                  <span class="rate-per">{{ tm('rateLimit.per') }}</span>
                  <v-text-field
                    v-model.number="limit.amount"
                    type="number"
                    min="1"
                    :label="tm('rateLimit.window')"
                    :error="!rateRowValid(limit)"
                    variant="outlined"
                    density="compact"
                    hide-details
                  />
                  <v-select
                    v-model="limit.unit"
                    :items="unitItems"
                    variant="outlined"
                    density="compact"
                    hide-details
                  />
                  <v-btn
                    icon="mdi-close"
                    size="small"
                    variant="text"
                    density="comfortable"
                    :title="tm('rules.delete')"
                    @click="rule.rate_limits.splice(li, 1)"
                  />
                </div>
                <div>
                  <v-btn
                    prepend-icon="mdi-plus"
                    variant="text"
                    color="primary"
                    size="small"
                    @click="rule.rate_limits.push({ count: 10, amount: 1, unit: 'm' })"
                  >
                    {{ tm('rateLimit.add') }}
                  </v-btn>
                </div>
                <div v-if="!rule.rate_limits.some(rateRowValid)" class="model-warning">
                  <v-icon size="16" color="warning">mdi-alert-outline</v-icon>
                  <span>{{ tm('rateLimit.errorEmpty') }}</span>
                </div>
                <div class="setting-subtitle">{{ tm('rateLimit.slidingHint') }}</div>
              </div>

              <div class="section-label">{{ tm('cronLimit.title') }}</div>
              <div class="dashboard-form-grid">
                <v-text-field
                  v-model="rule.cron_max_tasks"
                  type="number"
                  min="0"
                  :label="tm('cronLimit.maxTasks')"
                  :placeholder="tm('cronLimit.unset')"
                  :hint="tm('cronLimit.maxTasksHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
                <v-text-field
                  v-model="rule.cron_min_interval_hours"
                  type="number"
                  min="0"
                  :label="tm('cronLimit.interval')"
                  :placeholder="tm('cronLimit.unset')"
                  :hint="tm('cronLimit.intervalHint')"
                  persistent-hint
                  variant="outlined"
                  density="comfortable"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Tester -->
      <div class="dashboard-section-head">
        <div>
          <div class="dashboard-section-title">{{ tm('tester.title') }}</div>
          <div class="dashboard-section-subtitle">{{ tm('tester.subtitle') }}</div>
        </div>
      </div>

      <section class="dashboard-card dashboard-card--padded mb-5">
        <div class="tester-grid">
          <v-text-field
            v-model="tester.sender_id"
            :label="tm('tester.senderId')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
          <v-text-field
            v-model="tester.group_id"
            :label="tm('tester.groupId')"
            :placeholder="tm('tester.groupIdPlaceholder')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
          <v-select
            v-model="tester.role"
            :items="roleItems"
            :label="tm('tester.role')"
            variant="outlined"
            density="comfortable"
            hide-details
          />
        </div>

        <div class="tester-result mt-4">
          <template v-if="testResult">
            <div class="tester-verdict">
              <v-icon color="success">mdi-check-circle</v-icon>
              <span>
                {{ tm('tester.matched', { index: testResult.index + 1, name: ruleLabel(rules[testResult.index], testResult.index) }) }}
              </span>
              <span class="setting-subtitle mt-0">({{ tm('tester.matchedBy', { condition: testResult.condition }) }})</span>
            </div>
            <div class="dashboard-meta-list tester-policy">
              <div v-for="line in policyLines(rules[testResult.index])" :key="line.label" class="tester-policy-row">
                <span class="tester-policy-label">{{ line.label }}</span>
                <span class="tester-policy-value">{{ line.value }}</span>
              </div>
            </div>
          </template>
          <div v-else class="tester-verdict">
            <v-icon color="info">mdi-information-outline</v-icon>
            <span>{{ tm('tester.noMatch') }}</span>
          </div>

          <div v-if="rules.length" class="tester-trace mt-3">
            <div v-for="(status, idx) in testTrace" :key="rules[idx].__key" class="tester-trace-row">
              <v-chip size="x-small" label variant="tonal" :color="traceColor(status)">
                {{ tm(`tester.status.${status}`) }}
              </v-chip>
              <span>#{{ idx + 1 }} {{ ruleLabel(rules[idx], idx) }}</span>
            </div>
          </div>
        </div>
      </section>

      <v-snackbar v-model="snackbar.show" :color="snackbar.color" timeout="3000" location="top">
        {{ snackbar.message }}
        <template #actions>
          <v-btn variant="text" @click="snackbar.show = false">{{ tm('actions.close') }}</v-btn>
        </template>
      </v-snackbar>
    </v-container>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useTheme } from 'vuetify'
import { httpClient } from '@/api/http'
import { configProfileApi, mcpApi, personaApi, toolApi } from '@/api/v1'
import { useModuleI18n } from '@/i18n/composables'
import { askForConfirmation, useConfirmDialog } from '@/utils/confirmDialog'

type TriState = 'inherit' | 'allow' | 'deny'
type RateMode = 'inherit' | 'unlimited' | 'limit'
type Unit = 's' | 'm' | 'h' | 'd'
type RateRow = { count: number; amount: number; unit: Unit }
type PatternKey = 'tools_allow' | 'tools_deny' | 'mcp_allow' | 'mcp_deny'

type RuleRow = {
  __key: string
  __expanded: boolean
  // Keys this page does not know about, preserved on save.
  __extra: Record<string, unknown>
  id: string
  inherits: string
  name: string
  enabled: boolean
  match: string[]
  tools_allow: string[]
  tools_deny: string[]
  mcp_allow: string[]
  mcp_deny: string[]
  persona_id: string
  model: string
  global_memory: TriState
  rate_mode: RateMode
  rate_limits: RateRow[]
  rate_limit_reply: string
  // Scheduled-task limits; '' inherits (unset everywhere: the role's default).
  cron_max_tasks: string
  cron_min_interval_hours: string
}

type Facts = { sender_id: string; group_id: string; role: string }
type TraceStatus = 'matched' | 'shadowed' | 'noMatch' | 'disabled'

const CONFIG_ID = 'default'
const CONFIG_KEY = 'permission_rules'
const KNOWN_KEYS = [
  'id',
  'inherits',
  'rate_limit',
  'rate_limit_reply',
  'name',
  'enabled',
  'match',
  'tools_allow',
  'tools_deny',
  'mcp_allow',
  'mcp_deny',
  'persona_id',
  'model',
  'global_memory',
  'cron_max_tasks',
  'cron_min_interval_hours'
]

const { tm } = useModuleI18n('features/permissions')
const theme = useTheme()
const confirmDialog = useConfirmDialog()
const isDark = computed(() => theme.global.current.value.dark)

const loading = ref(false)
const saving = ref(false)
const hasLoaded = ref(false)
const initialSnapshot = ref('')
const rules = ref<RuleRow[]>([])
const snackbar = ref({ show: false, message: '', color: 'success' })

const toolNames = ref<string[]>([])
const toolsLoading = ref(false)
const mcpServerNames = ref<string[]>([])
const mcpLoading = ref(false)
const personaIds = ref<string[]>([])
const personasLoading = ref(false)
const models = ref<{ model?: string; id?: string; display_name?: string }[]>([])
const modelsLoading = ref(false)

// Mirrors MAX_INHERIT_DEPTH / MAX_WINDOW_S in astrbot/core/permission_rules.py.
const MAX_INHERIT_DEPTH = 16
const MAX_WINDOW_S = 30 * 24 * 3600
const UNIT_S: Record<Unit, number> = { s: 1, m: 60, h: 3600, d: 86400 }

const tester = ref<Facts>({ sender_id: '', group_id: '', role: 'member' })

const patternFields: { key: PatternKey; kind: 'tool' | 'mcp'; label: string; hint: string }[] = [
  { key: 'tools_allow', kind: 'tool', label: 'fields.toolsAllow', hint: 'fields.toolsAllowHint' },
  { key: 'tools_deny', kind: 'tool', label: 'fields.toolsDeny', hint: 'fields.toolsDenyHint' },
  { key: 'mcp_allow', kind: 'mcp', label: 'fields.mcpAllow', hint: 'fields.mcpAllowHint' },
  { key: 'mcp_deny', kind: 'mcp', label: 'fields.mcpDeny', hint: 'fields.mcpDenyHint' }
]

const triStateItems = computed(() => [
  { title: tm('triState.inherit'), value: 'inherit' },
  { title: tm('triState.allow'), value: 'allow' },
  { title: tm('triState.deny'), value: 'deny' }
])

const rateModeItems = computed(() => [
  { title: tm('rateLimit.inherit'), value: 'inherit' },
  { title: tm('rateLimit.unlimited'), value: 'unlimited' },
  { title: tm('rateLimit.limit'), value: 'limit' }
])

const unitItems = computed(() =>
  (Object.keys(UNIT_S) as Unit[]).map((u) => ({ title: tm(`rateLimit.units.${u}`), value: u }))
)

const roleItems = computed(() => [
  { title: tm('tester.roleMember'), value: 'member' },
  { title: tm('tester.roleAdmin'), value: 'admin' }
])

function toast(message: string, color: 'success' | 'error' | 'warning' = 'success') {
  snackbar.value = { show: true, message, color }
}

function errorMessage(e: any, fallback: string): string {
  if (typeof e === 'string') return e
  return e?.response?.data?.message || e?.message || fallback
}

function newKey(): string {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function newRuleId(): string {
  return `r_${Math.random().toString(16).slice(2, 10)}`
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : value == null ? '' : String(value)
}

// ---------- value coercion (mirrors astrbot/core/permission_rules.py) ----------

/** `_as_list`: a string is split on commas/newlines; a list is stringified. */
function asList(value: unknown): string[] {
  let items: string[]
  if (typeof value === 'string') items = value.replace(/,/g, '\n').split(/\r\n|\r|\n/)
  else if (Array.isArray(value)) items = value.map((v) => String(v))
  else items = []
  return items.map((i) => i.trim()).filter((i) => !!i)
}

/** `_as_opt_bool`: booleans, or the strings "true"/"false"; anything else is null. */
function asOptBool(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string' && ['true', 'false'].includes(value.toLowerCase())) {
    return value.toLowerCase() === 'true'
  }
  return null
}

function toTriState(value: unknown): TriState {
  const b = asOptBool(value)
  return b === null ? 'inherit' : b ? 'allow' : 'deny'
}

function fromTriState(value: TriState): boolean | null {
  return value === 'allow' ? true : value === 'deny' ? false : null
}

/** Combobox output → trimmed, de-duplicated list of non-empty strings. */
function cleanList(value: unknown): string[] {
  const list = Array.isArray(value) ? value : value == null ? [] : [value]
  const out: string[] = []
  for (const v of list) {
    const s = str(typeof v === 'object' && v !== null && 'value' in v ? (v as any).value : v).trim()
    if (s && !out.includes(s)) out.push(s)
  }
  return out
}

// ---------- matching (mirrors condition_matches / resolve_policy) ----------

function conditionMatches(condition: string, facts: Facts): boolean {
  const cond = condition.trim()
  if (!cond) return false
  if (cond === '*') return true
  if (cond.startsWith('role:')) return facts.role === cond.slice('role:'.length)
  if (cond.startsWith('p_')) return facts.sender_id === cond.slice(2)
  if (cond.startsWith('g_')) return !!facts.group_id && facts.group_id === cond.slice(2)
  if (cond.includes('/')) {
    const i = cond.indexOf('/')
    return facts.group_id === cond.slice(0, i) && facts.sender_id === cond.slice(i + 1)
  }
  return false
}

/** Returns an error message when the condition can never match, else ''. */
function conditionError(condition: string): string {
  const cond = condition.trim()
  if (!cond || cond === '*') return ''
  if (cond.startsWith('role:')) {
    const role = cond.slice('role:'.length)
    return role === 'admin' || role === 'member' ? '' : tm('match.errorRole', { condition: cond })
  }
  if (cond.startsWith('p_') || cond.startsWith('g_')) {
    return cond.length > 2 ? '' : tm('match.errorEmptyId', { condition: cond })
  }
  if (cond.includes('/')) {
    const i = cond.indexOf('/')
    return cond.slice(0, i) && cond.slice(i + 1) ? '' : tm('match.errorEmptyId', { condition: cond })
  }
  return tm('match.errorSyntax', { condition: cond })
}

function matchErrors(rule: RuleRow): string[] {
  const errs = rule.match.map(conditionError).filter((e) => !!e)
  // A rule without conditions is fine as a template other rules inherit.
  if (rule.enabled && rule.match.length === 0 && !isParent(rule)) errs.unshift(tm('match.errorNone'))
  return errs
}

// ---------- inheritance (mirrors inheritance_chain / policy_from_rule) ----------

function isParent(rule: RuleRow): boolean {
  return rules.value.some((r) => r !== rule && r.inherits === rule.id)
}

/** The rule followed by its ancestors; the first rule with an id owns it. */
function chainOf(rule: RuleRow): RuleRow[] {
  const chain = [rule]
  let current = rule
  while (chain.length < MAX_INHERIT_DEPTH) {
    const parent = rules.value.find((r) => r.id === current.inherits)
    if (!current.inherits || !parent || chain.includes(parent)) break
    chain.push(parent)
    current = parent
  }
  return chain
}

function parentItems(rule: RuleRow) {
  const items = [{ title: tm('fields.noParent'), value: '' }]
  rules.value.forEach((r, idx) => {
    // Leave out the rule itself and any rule that already inherits from it.
    if (r === rule || chainOf(r).includes(rule)) return
    items.push({ title: `#${idx + 1} ${ruleLabel(r, idx)}`, value: r.id })
  })
  if (rule.inherits && !rules.value.some((r) => r.id === rule.inherits)) {
    items.push({ title: tm('fields.parentMissing', { id: rule.inherits }), value: rule.inherits })
  }
  return items
}

// ---------- rate limits ----------

function rateRowValid(row: RateRow): boolean {
  const count = Number(row.count)
  const window = Number(row.amount) * UNIT_S[row.unit]
  return Number.isInteger(count) && count >= 1 && Number.isInteger(window) && window >= 1 && window <= MAX_WINDOW_S
}

function toRateRow(window: number, count: number): RateRow {
  const unit = (['d', 'h', 'm'] as Unit[]).find((u) => window % UNIT_S[u] === 0) || 's'
  return { count, amount: window / UNIT_S[unit], unit }
}

/** '' when the rule inherits its rate limits, else a readable description. */
function rateText(rule: RuleRow): string {
  if (rule.rate_mode === 'inherit') return ''
  const rows = rule.rate_mode === 'limit' ? rule.rate_limits.filter(rateRowValid) : []
  if (!rows.length) return tm('rateLimit.unlimited')
  return rows
    .map((r) => tm('rateLimit.item', { count: r.count, amount: r.amount, unit: tm(`rateLimit.units.${r.unit}`) }))
    .join(', ')
}

const normalizedFacts = computed<Facts>(() => ({
  sender_id: tester.value.sender_id.trim(),
  group_id: tester.value.group_id.trim(),
  role: tester.value.role || 'member'
}))

const testTrace = computed<TraceStatus[]>(() => {
  let found = false
  return rules.value.map((rule) => {
    if (!rule.enabled) return 'disabled'
    const hit = rule.match.some((c) => conditionMatches(c, normalizedFacts.value))
    if (!hit) return 'noMatch'
    if (found) return 'shadowed'
    found = true
    return 'matched'
  })
})

const testResult = computed(() => {
  const index = testTrace.value.indexOf('matched')
  if (index < 0) return null
  const condition = rules.value[index].match.find((c) => conditionMatches(c, normalizedFacts.value)) || ''
  return { index, condition }
})

function traceColor(status: TraceStatus): string {
  if (status === 'matched') return 'success'
  if (status === 'shadowed') return 'warning'
  return 'grey'
}

function ruleLabel(rule: RuleRow | undefined, idx: number): string {
  return rule?.name.trim() || tm('rules.unnamed', { index: idx + 1 })
}

function triLabel(value: TriState): string {
  return tm(`triState.${value}`)
}

/** The effective policy: each field from the nearest rule in the chain that sets it. */
function policyLines(rule: RuleRow | undefined): { label: string; value: string }[] {
  if (!rule) return []
  const chain = chainOf(rule)
  const nearest = (pick: (r: RuleRow) => string, unset: string) => {
    for (const r of chain) {
      const value = pick(r)
      if (!value) continue
      if (r === rule) return value
      return `${value} ${tm('tester.inheritedFrom', { name: ruleLabel(r, rules.value.indexOf(r)) })}`
    }
    return unset
  }
  const list = (key: PatternKey) => nearest((r) => r[key].join(', '), '—')
  const tri = (key: 'global_memory') =>
    nearest((r) => (r[key] === 'inherit' ? '' : triLabel(r[key])), triLabel('inherit'))
  return [
    { label: tm('tester.chain'), value: chain.map((r) => ruleLabel(r, rules.value.indexOf(r))).join(' → ') },
    { label: tm('fields.toolsAllow'), value: list('tools_allow') },
    { label: tm('fields.toolsDeny'), value: list('tools_deny') },
    { label: tm('fields.mcpAllow'), value: list('mcp_allow') },
    { label: tm('fields.mcpDeny'), value: list('mcp_deny') },
    { label: tm('fields.persona'), value: nearest((r) => r.persona_id, tm('fields.noOverride')) },
    { label: tm('fields.model'), value: nearest((r) => r.model, tm('fields.noOverride')) },
    { label: tm('fields.globalMemory'), value: tri('global_memory') },
    { label: tm('rateLimit.title'), value: nearest(rateText, tm('rateLimit.unlimited')) },
    {
      label: tm('rateLimit.reply'),
      value: nearest((r) => r.rate_limit_reply.trim(), tm('rateLimit.replyPlaceholder'))
    },
    { label: tm('cronLimit.maxTasks'), value: nearest((r) => cronText(r.cron_max_tasks), tm('cronLimit.unset')) },
    {
      label: tm('cronLimit.interval'),
      value: nearest((r) => cronText(r.cron_min_interval_hours), tm('cronLimit.unset'))
    }
  ]
}

function ruleSummary(rule: RuleRow): string {
  const parts: string[] = []
  if (rule.inherits) {
    const parent = rules.value.findIndex((r) => r.id === rule.inherits)
    const name = parent >= 0 ? ruleLabel(rules.value[parent], parent) : rule.inherits
    parts.push(`${tm('fields.inherits')}: ${name}`)
  }
  if (rule.tools_allow.length || rule.tools_deny.length) parts.push(tm('fields.toolsTitle'))
  if (rule.mcp_allow.length || rule.mcp_deny.length) parts.push('MCP')
  if (rule.persona_id) parts.push(`${tm('fields.persona')}: ${rule.persona_id}`)
  if (rule.model) parts.push(`${tm('fields.model')}: ${rule.model}`)
  if (rule.global_memory !== 'inherit') parts.push(`${tm('fields.globalMemory')}: ${triLabel(rule.global_memory)}`)
  if (rateText(rule)) parts.push(`${tm('rateLimit.title')}: ${rateText(rule)}`)
  if (cronText(rule.cron_max_tasks)) parts.push(`${tm('cronLimit.maxTasks')}: ${cronText(rule.cron_max_tasks)}`)
  if (cronText(rule.cron_min_interval_hours)) {
    parts.push(`${tm('cronLimit.interval')}: ${cronText(rule.cron_min_interval_hours)}`)
  }
  return parts.join(' · ')
}

// ---------- rule list editing ----------

function emptyRule(): RuleRow {
  return {
    __key: newKey(),
    __expanded: true,
    __extra: {},
    id: newRuleId(),
    inherits: '',
    name: '',
    enabled: true,
    match: [],
    tools_allow: [],
    tools_deny: [],
    mcp_allow: [],
    mcp_deny: [],
    persona_id: '',
    model: '',
    global_memory: 'inherit',
    rate_mode: 'inherit',
    rate_limits: [],
    rate_limit_reply: '',
    cron_max_tasks: '',
    cron_min_interval_hours: ''
  }
}

function addRule() {
  rules.value.push(emptyRule())
}

function moveRule(idx: number, delta: number) {
  const target = idx + delta
  if (target < 0 || target >= rules.value.length) return
  const [row] = rules.value.splice(idx, 1)
  rules.value.splice(target, 0, row)
}

function duplicateRule(idx: number) {
  const src = rules.value[idx]
  const copy: RuleRow = {
    ...JSON.parse(JSON.stringify(src)),
    __key: newKey(),
    __expanded: true,
    id: newRuleId(),
    name: src.name ? tm('rules.copyName', { name: src.name }) : ''
  }
  rules.value.splice(idx + 1, 0, copy)
}

async function deleteRule(idx: number) {
  const rule = rules.value[idx]
  const name = ruleLabel(rule, idx)
  const children = rules.value.filter((r) => r !== rule && r.inherits === rule.id)
  const confirmed = await askForConfirmation(
    children.length
      ? tm('messages.deleteParentConfirm', { name, count: children.length })
      : tm('messages.deleteConfirm', { name }),
    confirmDialog
  )
  if (!confirmed) return
  // Its children now inherit from its own parent.
  for (const child of children) child.inherits = rule.inherits
  rules.value.splice(idx, 1)
}

// ---------- suggestions ----------

async function loadTools() {
  toolsLoading.value = true
  try {
    const res = await toolApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    toolNames.value = Array.from(new Set(data.map((t: any) => str(t?.name)).filter((n: string) => !!n))).sort()
  } catch {
    toolNames.value = []
  } finally {
    toolsLoading.value = false
  }
}

async function loadMcpServers() {
  mcpLoading.value = true
  try {
    const res = await mcpApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    mcpServerNames.value = Array.from(new Set(data.map((s: any) => str(s?.name)).filter((n: string) => !!n))).sort()
  } catch {
    mcpServerNames.value = []
  } finally {
    mcpLoading.value = false
  }
}

async function loadPersonas() {
  personasLoading.value = true
  try {
    const res = await personaApi.list()
    const data = res.data?.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
    personaIds.value = Array.from(new Set(data.map((p: any) => str(p?.persona_id)).filter((n: string) => !!n)))
  } catch {
    personaIds.value = []
  } finally {
    personasLoading.value = false
  }
}

async function loadModels() {
  modelsLoading.value = true
  try {
    const res = await httpClient.get('/api/codex/models', { params: { include_hidden: false } })
    models.value = res.data.status === 'ok' && Array.isArray(res.data.data) ? res.data.data : []
  } catch {
    models.value = []
  } finally {
    modelsLoading.value = false
  }
}

const modelIds = computed(() => models.value.map((m) => str(m.model || m.id)).filter((id) => !!id))

function modelDisplayName(id: string): string {
  const info = models.value.find((m) => (m.model || m.id) === id)
  return info?.display_name || id
}

function personaItems(current: string) {
  const items = [{ title: tm('fields.noOverride'), value: '' }]
  for (const id of personaIds.value) items.push({ title: id, value: id })
  if (current && !personaIds.value.includes(current)) {
    items.push({ title: tm('fields.personaMissing', { id: current }), value: current })
  }
  return items
}

// ---------- config load/save ----------

/** A scheduled-task limit as the backend reads it: a number of 0 or more. */
function cronValid(value: unknown): boolean {
  const text = String(value ?? '').trim()
  return text !== '' && Number.isFinite(Number(text)) && Number(text) >= 0
}

function cronValue(raw: unknown): string {
  return cronValid(raw) ? String(Number(raw)) : ''
}

/** A limit set on a rule, for display ('' when it inherits). */
function cronText(value: string): string {
  if (!cronValid(value)) return ''
  return Number(value) === 0 ? tm('cronLimit.unlimited') : String(Number(value))
}

function normalizeRule(raw: Record<string, unknown>): RuleRow {
  const extra: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(raw)) {
    if (!KNOWN_KEYS.includes(k)) extra[k] = v
  }
  // Like parse_rate_limits: a list sets limits (one left empty means none).
  const limits = Array.isArray(raw.rate_limit)
    ? raw.rate_limit
        .map((l: any) => ({ window: Math.trunc(Number(l?.window)), count: Math.trunc(Number(l?.count)) }))
        .filter((l) => l.window > 0 && l.window <= MAX_WINDOW_S && l.count > 0)
    : []
  let rateMode: RateMode = 'unlimited'
  if (raw.rate_limit == null || raw.rate_limit === 'inherit') rateMode = 'inherit'
  else if (limits.length) rateMode = 'limit'
  return {
    __key: newKey(),
    __expanded: false,
    __extra: extra,
    id: str(raw.id).trim(),
    inherits: str(raw.inherits).trim(),
    name: str(raw.name),
    // Only an explicit false (or "false") disables a rule, like the backend.
    enabled: asOptBool(raw.enabled === undefined ? true : raw.enabled) !== false,
    match: asList(raw.match),
    tools_allow: asList(raw.tools_allow),
    tools_deny: asList(raw.tools_deny),
    mcp_allow: asList(raw.mcp_allow),
    mcp_deny: asList(raw.mcp_deny),
    persona_id: str(raw.persona_id),
    model: str(raw.model),
    global_memory: toTriState(raw.global_memory),
    rate_mode: rateMode,
    rate_limits: limits.map((l) => toRateRow(l.window, l.count)),
    rate_limit_reply: str(raw.rate_limit_reply),
    cron_max_tasks: cronValue(raw.cron_max_tasks),
    cron_min_interval_hours: cronValue(raw.cron_min_interval_hours)
  }
}

function normalizeRules(value: unknown): RuleRow[] {
  if (!Array.isArray(value)) return []
  const rows = value
    .filter((r): r is Record<string, unknown> => !!r && typeof r === 'object' && !Array.isArray(r))
    .map(normalizeRule)
  // Every rule gets an id to be inherited by; a later duplicate gets a new one
  // (the first rule with an id owns it, as in the backend).
  const seen = new Set<string>()
  for (const row of rows) {
    if (!row.id || seen.has(row.id)) row.id = newRuleId()
    seen.add(row.id)
  }
  return rows
}

function rulesPayload(rows: RuleRow[]) {
  return rows.map((r) => ({
    ...r.__extra,
    id: r.id,
    inherits: r.inherits,
    name: r.name.trim(),
    enabled: !!r.enabled,
    match: cleanList(r.match),
    tools_allow: cleanList(r.tools_allow),
    tools_deny: cleanList(r.tools_deny),
    mcp_allow: cleanList(r.mcp_allow),
    mcp_deny: cleanList(r.mcp_deny),
    persona_id: r.persona_id || '',
    model: (r.model || '').trim(),
    global_memory: fromTriState(r.global_memory),
    rate_limit:
      r.rate_mode === 'limit'
        ? r.rate_limits
            .filter(rateRowValid)
            .map((l) => ({ window: Number(l.amount) * UNIT_S[l.unit], count: Number(l.count) }))
        : r.rate_mode,
    rate_limit_reply: r.rate_limit_reply.trim(),
    cron_max_tasks: cronValid(r.cron_max_tasks) ? Number(r.cron_max_tasks) : 'inherit',
    cron_min_interval_hours: cronValid(r.cron_min_interval_hours)
      ? Number(r.cron_min_interval_hours)
      : 'inherit'
  }))
}

const hasUnsavedChanges = computed(
  () => hasLoaded.value && JSON.stringify(rulesPayload(rules.value)) !== initialSnapshot.value
)

async function fetchDefaultConfig(): Promise<any> {
  const res = await configProfileApi.get(CONFIG_ID)
  if (res.data.status !== 'ok') {
    throw new Error(res.data.message || tm('messages.loadConfigFailed'))
  }
  return (res.data.data as any)?.config || {}
}

async function loadConfig() {
  try {
    const config = await fetchDefaultConfig()
    rules.value = normalizeRules(config?.[CONFIG_KEY])
    initialSnapshot.value = JSON.stringify(rulesPayload(rules.value))
    hasLoaded.value = true
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.loadConfigFailed')), 'error')
  }
}

async function confirmProblems(): Promise<boolean> {
  const problems: string[] = []
  rules.value.forEach((rule, idx) => {
    // A disabled rule still matters as a template others inherit.
    if (!rule.enabled && !isParent(rule)) return
    const errs = rule.enabled ? matchErrors(rule) : []
    if (rule.rate_mode === 'limit') {
      if (!rule.rate_limits.every(rateRowValid)) errs.push(tm('rateLimit.errorRow'))
      if (!rule.rate_limits.some(rateRowValid)) errs.push(tm('rateLimit.errorEmpty'))
    }
    const tasks = String(rule.cron_max_tasks).trim()
    const hours = String(rule.cron_min_interval_hours).trim()
    if ((tasks && !(cronValid(tasks) && Number.isInteger(Number(tasks)))) || (hours && !cronValid(hours))) {
      errs.push(tm('cronLimit.errorNumber'))
    }
    if (errs.length) problems.push(`#${idx + 1} ${ruleLabel(rule, idx)}: ${errs.join('; ')}`)
  })
  if (!problems.length) return true
  return askForConfirmation(
    `${tm('messages.invalidConditionsConfirm')}\n\n${problems.join('\n')}`,
    confirmDialog
  )
}

async function save() {
  if (!(await confirmProblems())) return
  saving.value = true
  try {
    // Re-fetch so that only permission_rules is replaced and every other
    // config field keeps its latest stored value.
    const config = await fetchDefaultConfig()
    const payload = rulesPayload(rules.value)
    config[CONFIG_KEY] = payload
    const res = await configProfileApi.update(CONFIG_ID, config)
    if (res.data.status === 'ok') {
      initialSnapshot.value = JSON.stringify(payload)
      toast(res.data.message || tm('messages.saveSuccess'))
    } else {
      toast(res.data.message || tm('messages.saveFailed'), 'error')
    }
  } catch (e: any) {
    toast(errorMessage(e, tm('messages.saveFailed')), 'error')
  } finally {
    saving.value = false
  }
}

async function reload() {
  if (hasUnsavedChanges.value) {
    const confirmed = await askForConfirmation(tm('messages.unsavedChangesReloadConfirm'), confirmDialog)
    if (!confirmed) return
  }
  loading.value = true
  try {
    await Promise.all([loadConfig(), loadTools(), loadMcpServers(), loadPersonas(), loadModels()])
  } finally {
    loading.value = false
  }
}

function handleBeforeUnload(event: BeforeUnloadEvent) {
  if (!hasUnsavedChanges.value) return
  event.preventDefault()
  event.returnValue = ''
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload)
  reload()
})

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload)
})

onBeforeRouteLeave(async () => {
  if (!hasUnsavedChanges.value) return true
  return askForConfirmation(tm('messages.unsavedChangesLeaveConfirm'), confirmDialog)
})
</script>

<style scoped>
@import '@/styles/dashboard-shell.css';

.permission-rules-page {
  padding-bottom: 40px;
}

.unsaved-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  margin-bottom: 18px;
  border: 1px solid rgba(var(--v-theme-warning), 0.22);
  border-radius: 12px;
  background: rgba(var(--v-theme-warning), 0.08);
  color: var(--dashboard-text);
  font-size: 13px;
  line-height: 1.5;
}

.syntax-help {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}

.syntax-list {
  margin: 4px 0 0 18px;
}

.syntax-list code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--dashboard-text);
}

.rule-list {
  display: grid;
  gap: 14px;
}

.rule-card {
  border: 1px solid var(--dashboard-border);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(var(--v-theme-primary), 0.02);
  min-width: 0;
}

.rule-card--disabled {
  opacity: 0.65;
}

.rule-card--hit {
  border-color: rgba(var(--v-theme-success), 0.6);
}

.rule-head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.rule-index {
  font-weight: 700;
  color: var(--dashboard-muted);
  min-width: 28px;
}

.rule-name {
  flex: 1 1 240px;
  min-width: 180px;
}

.rule-switch {
  flex: 0 0 auto;
}

.rule-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

.rule-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.rule-body {
  margin-top: 14px;
}

.section-label {
  margin: 18px 0 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--dashboard-muted);
}

.setting-subtitle {
  margin-top: 6px;
  color: var(--dashboard-muted);
  font-size: 13px;
  line-height: 1.6;
}


.model-warning {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  font-size: 13px;
  color: rgb(var(--v-theme-warning));
}

.rate-list {
  display: grid;
  gap: 8px;
}

.rate-row {
  display: grid;
  grid-template-columns: 120px auto 120px 120px auto;
  align-items: center;
  gap: 8px;
  max-width: 560px;
}

.rate-per {
  color: var(--dashboard-muted);
  font-size: 13px;
}

.tester-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 200px;
  gap: 12px;
}

.tester-verdict {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-weight: 600;
}

.tester-policy {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 18px;
  margin-top: 10px;
}

.tester-policy-row {
  display: flex;
  gap: 8px;
  font-size: 13px;
  min-width: 0;
}

.tester-policy-label {
  color: var(--dashboard-muted);
  min-width: 140px;
}

.tester-policy-value {
  word-break: break-all;
}

.tester-trace {
  display: grid;
  gap: 4px;
  font-size: 13px;
}

.tester-trace-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 900px) {
  .tester-grid,
  .tester-policy {
    grid-template-columns: 1fr;
  }

  .rate-row {
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  }
}
</style>
