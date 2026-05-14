import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml'
import {
  api,
  type ChatHistoryRow,
  type ModelRow,
  type ConfigPayload,
  type QueryResult,
  type ExampleRow,
  type StoredConfig,
  type TableRow,
  type ColumnRow,
  type KnowledgeRow,
  toPayload,
} from './api'
import './App.css'

type View = 'settings' | 'chat' | 'validator' | 'metadata' | 'search-knowledge'

type Status = {
  kind: 'success' | 'error' | 'info'
  message: string
} | null

type ConversationItem = {
  id: string
  firstUserMessage: string
  userMessage: ChatHistoryRow | null
  assistantMessage: ChatHistoryRow | null
}

type ResponseSegment =
  | { kind: 'text'; content: string }
  | { kind: 'code'; content: string; language: string }

const emptyConfig: ConfigPayload = {
  database_server: {
    hostname: 'localhost',
    port: 5432,
    database: '',
    user: 'postgres',
    password: '',
    type: 'postgresql',
    schema: 'public',
    title: 'Untitled',
    url: '',
  },
  chat_database_server: {
    hostname: 'localhost',
    port: 5432,
    database: '',
    user: 'postgres',
    password: '',
    schema: 'public',
  },
}

type UnknownRecord = Record<string, unknown>

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function firstDefined(record: UnknownRecord, keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) {
      return record[key]
    }
  }
  return undefined
}

function asNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function hasAnyKey(record: UnknownRecord, keys: string[]): boolean {
  return keys.some((key) => key in record)
}

function extractUploadedConfigCandidate(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value[0]
  }

  if (!isRecord(value)) return value

  const wrapped = value.config ?? value.configuration ?? value.payload ?? value.data ?? value.active ?? value.selected_config
  if (wrapped !== undefined) return extractUploadedConfigCandidate(wrapped)

  if (Array.isArray(value.items)) return extractUploadedConfigCandidate(value.items)
  if (Array.isArray(value.configs)) return extractUploadedConfigCandidate(value.configs)

  return value
}

function normalizeUploadedConfig(value: unknown): ConfigPayload | null {
  const candidate = extractUploadedConfigCandidate(value)
  if (!isRecord(candidate)) return null

  const dbRecord = isRecord(candidate.database_server)
    ? candidate.database_server
    : isRecord(candidate.db_cfg)
      ? candidate.db_cfg
      : candidate
  const chatRecord = isRecord(candidate.chat_database_server)
    ? candidate.chat_database_server
    : isRecord(candidate.chat_cfg)
      ? candidate.chat_cfg
      : candidate

  const hasDatabaseShape = hasAnyKey(dbRecord, [
    'hostname',
    'host',
    'db_hostname',
    'url',
    'db_url',
    'connection_string',
    'connectionstring',
    'conn_string',
    'database',
    'db_database',
  ])
  const hasChatShape = hasAnyKey(chatRecord, [
    'hostname',
    'host',
    'chat_hostname',
    'database',
    'chat_database',
    'user',
    'chat_user',
  ])

  if (!hasDatabaseShape && !hasChatShape) {
    return null
  }

  return {
    database_server: {
      hostname: asString(firstDefined(dbRecord, ['hostname', 'host', 'server', 'db_hostname']), emptyConfig.database_server.hostname ?? 'localhost'),
      port: asNumber(firstDefined(dbRecord, ['port', 'db_port']), emptyConfig.database_server.port ?? 5432),
      database: asString(firstDefined(dbRecord, ['database', 'dbname', 'name', 'db_database'])),
      user: asString(firstDefined(dbRecord, ['user', 'username', 'db_user']), emptyConfig.database_server.user ?? 'postgres'),
      password: asString(firstDefined(dbRecord, ['password', 'pass', 'db_password'])),
      type: asString(firstDefined(dbRecord, ['type', 'dialect', 'engine', 'db_type']), emptyConfig.database_server.type ?? 'postgresql'),
      schema: asString(firstDefined(dbRecord, ['schema', 'schema_name', 'db_schema']), emptyConfig.database_server.schema ?? 'public'),
      title: asString(firstDefined(dbRecord, ['title', 'db_title', 'label']), emptyConfig.database_server.title ?? 'Untitled'),
      url: asString(firstDefined(dbRecord, ['url', 'db_url', 'connection_string', 'connectionstring', 'conn_string', 'connectionUrl', 'connection_url'])),
    },
    chat_database_server: {
      hostname: asString(firstDefined(chatRecord, ['hostname', 'host', 'server', 'chat_hostname']), emptyConfig.chat_database_server.hostname ?? 'localhost'),
      port: asNumber(firstDefined(chatRecord, ['port', 'chat_port']), emptyConfig.chat_database_server.port ?? 5432),
      database: asString(firstDefined(chatRecord, ['database', 'dbname', 'name', 'chat_database'])),
      user: asString(firstDefined(chatRecord, ['user', 'username', 'chat_user']), emptyConfig.chat_database_server.user ?? 'postgres'),
      password: asString(firstDefined(chatRecord, ['password', 'pass', 'chat_password'])),
      schema: asString(firstDefined(chatRecord, ['schema', 'schema_name', 'chat_schema']), emptyConfig.chat_database_server.schema ?? 'public'),
    },
  }
}

function parseUploadedConfigText(fileName: string, text: string): unknown {
  const lowerName = fileName.toLowerCase()

  if (lowerName.endsWith('.yaml') || lowerName.endsWith('.yml')) {
    return parseYaml(text)
  }

  return JSON.parse(text)
}

function isUserRole(role: string): boolean {
  const normalized = role.toLowerCase()
  return normalized === 'user' || normalized === 'human'
}

function shortPreview(text: string, maxLen = 90): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  if (clean.length <= maxLen) return clean
  return `${clean.slice(0, maxLen)}...`
}

function splitResponseIntoSegments(content: string): ResponseSegment[] {
  const segments: ResponseSegment[] = []
  const regex = /```(\w+)?\n([\s\S]*?)```/g

  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(content)) !== null) {
    const textBefore = content.slice(lastIndex, match.index)
    if (textBefore.trim()) {
      segments.push({ kind: 'text', content: textBefore.trim() })
    }
    segments.push({
      kind: 'code',
      language: (match[1] ?? 'code').toLowerCase(),
      content: match[2].trim(),
    })
    lastIndex = regex.lastIndex
  }

  const tail = content.slice(lastIndex)
  if (tail.trim()) {
    segments.push({ kind: 'text', content: tail.trim() })
  }

  if (segments.length === 0 && content.trim()) {
    segments.push({ kind: 'text', content: content.trim() })
  }

  return segments
}

type StreamChunk =
  | { type: 'text'; content: string }
  | { type: 'code'; content: string; language: string }

function parseStreamChunks(content: string): StreamChunk[] {
  if (!content.trim()) return []

  const chunks: StreamChunk[] = []
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g
  let lastIndex = 0

  for (const match of content.matchAll(codeBlockRegex)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      const textChunk = content.slice(lastIndex, index).trim()
      if (textChunk) {
        chunks.push({ type: 'text', content: textChunk })
      }
    }

    const language = (match[1] || '').trim().toLowerCase() || 'text'
    const code = (match[2] || '').trim()
    if (code) {
      chunks.push({ type: 'code', language, content: code })
    }
    lastIndex = index + match[0].length
  }

  if (lastIndex < content.length) {
    const trailing = content.slice(lastIndex).trim()
    if (trailing) {
      chunks.push({ type: 'text', content: trailing })
    }
  }

  return chunks.length > 0 ? chunks : [{ type: 'text', content }]
}

function createRandomUnusedSessionId(usedIds: Set<number>): number {
  // Try random 6-digit values first to avoid predictable collisions.
  for (let i = 0; i < 64; i += 1) {
    const candidate = Math.floor(100000 + Math.random() * 900000)
    if (!usedIds.has(candidate)) return candidate
  }

  // Deterministic fallback if random attempts are unlucky.
  let candidate = 100000
  while (usedIds.has(candidate)) {
    candidate += 1
  }
  return candidate
}

function onEnterPress(
  event: React.KeyboardEvent<HTMLInputElement>,
  action: () => void,
) {
  if (event.key !== 'Enter') return
  event.preventDefault()
  action()
}

function makeAutoSessionLabel(prefix: string): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  return `${prefix} ${stamp}`
}

function App() {
  const [view, setView] = useState<View>('settings')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('app-theme')
    return (saved as 'light' | 'dark') || 'light'
  })
  const [status, setStatus] = useState<Status>(null)
  const [lastSqlError, setLastSqlError] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const [configs, setConfigs] = useState<StoredConfig[]>([])
  const [selectedConfigId, setSelectedConfigId] = useState<number | ''>('')
  const [form, setForm] = useState<ConfigPayload>(emptyConfig)
  const [activeConfig, setActiveConfig] = useState<(ConfigPayload & { id?: number }) | null>(null)

  const [examples, setExamples] = useState<ExampleRow[]>([])
  const [tables, setTables] = useState<TableRow[]>([])
  const [columns, setColumns] = useState<ColumnRow[]>([])
  const [knowledge, setKnowledge] = useState<KnowledgeRow[]>([])
  const [knowledgeReference, setKnowledgeReference] = useState('')
  const [knowledgeContent, setKnowledgeContent] = useState('')
  const [editingKnowledgeId, setEditingKnowledgeId] = useState<number | null>(null)
  const [editingKnowledgeReference, setEditingKnowledgeReference] = useState('')
  const [vectorStats, setVectorStats] = useState<Record<string, number>>({})
  const [vectorQuery, setVectorQuery] = useState('')
  const [vectorCollectionsInput, setVectorCollectionsInput] = useState('')
  const [vectorK, setVectorK] = useState(5)
  const [vectorResults, setVectorResults] = useState<Record<string, Array<{ content: string; metadata: Record<string, unknown>; score: number }>>>({})

  const [sessionId, setSessionId] = useState<number | ''>('')
  const [pastSessions, setPastSessions] = useState<Array<{ id: number; label: string; timestamp: string }>>([])  
  const [sessionForm, setSessionForm] = useState({
    username: 'demo-user',
    email: 'demo@example.com',
    label: '',
    meta: '',
  })
  const [history, setHistory] = useState<ChatHistoryRow[]>([])
  const [models, setModels] = useState<ModelRow[]>([])
  const [selectedModel, setSelectedModel] = useState('api')
  const [selectedModelProvider, setSelectedModelProvider] = useState<string | null>(null)
  const [newModelName, setNewModelName] = useState('')
  const [newModelProvider, setNewModelProvider] = useState('')
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatSql, setChatSql] = useState('select 1 as value')
  const [plottingInstructions, setPlottingInstructions] = useState('')
  const [agentResponse, setAgentResponse] = useState('')
  const [editableAgentSql, setEditableAgentSql] = useState<Record<number, string>>({})
  const [codeRunResults, setCodeRunResults] = useState<Map<number, { plot_html: string | null; stdout: string; stderr: string; error: string | null }>>(new Map())
  const [pendingValidationQuestion, setPendingValidationQuestion] = useState('')
  const [showValidationConfirm, setShowValidationConfirm] = useState(false)
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryResultSource, setQueryResultSource] = useState<string | null>(null)
  const [pendingUploadConnect, setPendingUploadConnect] = useState(false)
  const [uploadedConfigFileName, setUploadedConfigFileName] = useState('')
  const configUploadRef = useRef<HTMLInputElement | null>(null)
  const parsedAgentResponse = useMemo(() => parseStreamChunks(agentResponse), [agentResponse])

  // Apply theme to document
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
    }
    localStorage.setItem('app-theme', theme)
  }, [theme])

  useEffect(() => {
    if (view !== 'validator') return
    void loadExamples()
  }, [view])

  // On mount: load saved configs and restore active config from backend
  useEffect(() => {
    async function init() {
      try {
        const [configsData, activeData, modelsData, sessionsData] = await Promise.all([
          api.listConfigs(),
          api.getActive(),
          api.listModels(),
          api.listSessions().catch(() => ({ items: [] })),
        ])
        setConfigs(configsData.items)
        setModels(modelsData.items)
        if (modelsData.items.length > 0) {
          setSelectedModel(modelsData.items[0].model_name)
          setSelectedModelProvider(modelsData.items[0].provider ?? null)
        }
        if (sessionsData.items.length > 0) {
          setPastSessions(sessionsData.items.map((s) => ({
            id: s.session_id,
            label: s.content ? s.content.slice(0, 60).replace(/\s+/g, ' ').trim() : `Session ${s.session_id}`,
            timestamp: s.timestamp ?? '',
          })))
        }
        // Create a real persisted session on startup so saves always target a valid session id.
        const bootSession = await api.createSession({
          ...sessionForm,
          label: sessionForm.label.trim() || makeAutoSessionLabel('Boot session'),
          referenced_session_id: null,
        })
        if (bootSession?.session_id) {
          setSessionId(bootSession.session_id)
          setPastSessions((prev) => {
            if (prev.some((s) => s.id === bootSession.session_id)) return prev
            const label = sessionForm.label.trim() || `Session ${bootSession.session_id}`
            return [{ id: bootSession.session_id, label, timestamp: '' }, ...prev]
          })
          try {
            const historyResponse = await api.getSessionHistory(bootSession.session_id)
            setHistory(historyResponse.items)
          } catch {
            setHistory([])
          }
        } else {
          const usedSessionIds = new Set<number>(sessionsData.items.map((s) => s.session_id))
          setSessionId(createRandomUnusedSessionId(usedSessionIds))
        }
        if (activeData.active) {
          setActiveConfig(activeData.active)
          setForm(activeData.active)
          if (activeData.active.id !== undefined) {
            setSelectedConfigId(activeData.active.id)
          }
        }
      } catch {
        // Silently ignore init failures (backend may not be ready)
      }
    }
    init()
  }, [])

  const activeSummary = useMemo(() => {
    if (!activeConfig) return 'No active backend configuration'
    const db = activeConfig.database_server
    return `${db.title ?? 'Untitled'} (${db.hostname}:${db.port}/${db.database})`
  }, [activeConfig])

  const conversations = useMemo<ConversationItem[]>(() => {
    const rows = [...history].sort((a, b) => a.sequence - b.sequence || a.id - b.id)
    const grouped: ConversationItem[] = []
    let pendingUser: ChatHistoryRow | null = null

    for (const row of rows) {
      if (isUserRole(row.role)) {
        pendingUser = row
        continue
      }

      if (pendingUser) {
        grouped.push({
          id: `conv-${pendingUser.id}-${row.id}`,
          firstUserMessage: pendingUser.content,
          userMessage: pendingUser,
          assistantMessage: row,
        })
        pendingUser = null
      } else {
        grouped.push({
          id: `conv-orphan-${row.id}`,
          firstUserMessage: row.content,
          userMessage: null,
          assistantMessage: row,
        })
      }
    }

    if (pendingUser) {
      grouped.push({
        id: `conv-pending-${pendingUser.id}`,
        firstUserMessage: pendingUser.content,
        userMessage: pendingUser,
        assistantMessage: null,
      })
    }

    return grouped
  }, [history])

  const hasSqlAlchemyError = useMemo(() => {
    const text = lastSqlError.toLowerCase()
    if (!text) return false
    return text.includes('sqlalchemy') || text.includes('psycopg2') || text.includes('pymssql')
  }, [lastSqlError])

  useEffect(() => {
    const textareas = document.querySelectorAll<HTMLTextAreaElement>('textarea.agent-sql-editor')
    textareas.forEach((textarea) => {
      textarea.style.height = 'auto'
      textarea.style.height = `${textarea.scrollHeight}px`
    })
  }, [editableAgentSql, parsedAgentResponse])

  const sqlCorrectionHistorySummary = useMemo(() => {
    const tail = conversations.slice(-6)
    if (tail.length === 0) return ''
    return tail
      .map((conv, idx) => {
        const userText = (conv.userMessage?.content ?? conv.firstUserMessage ?? '').replace(/\s+/g, ' ').trim()
        const assistantText = (conv.assistantMessage?.content ?? '').replace(/\s+/g, ' ').trim()
        const userShort = userText.length > 240 ? `${userText.slice(0, 240)}...` : userText
        const assistantShort = assistantText.length > 240 ? `${assistantText.slice(0, 240)}...` : assistantText
        return `Turn ${idx + 1}:\nUser: ${userShort || '(empty)'}\nAssistant: ${assistantShort || '(empty)'}`
      })
      .join('\n\n')
  }, [conversations])

  async function run<T>(fn: () => Promise<T>, successMsg?: string): Promise<T | undefined> {
    setLoading(true)
    setStatus(null)
    try {
      const data = await fn()
      if (successMsg) setStatus({ kind: 'success', message: successMsg })
      return data
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus({ kind: 'error', message })
      return undefined
    } finally {
      setLoading(false)
    }
  }

  async function refreshConfigs() {
    const data = await run(() => api.listConfigs())
    if (!data) return
    setConfigs(data.items)
  }

  async function refreshActiveConfig() {
    const data = await run(() => api.getActive())
    if (!data) return
    setActiveConfig(data.active)
    if (data.active) setForm(data.active)
  }

  async function loadFromSelected() {
    if (selectedConfigId === '') {
      setStatus({ kind: 'info', message: 'Select a config first.' })
      return
    }

    const selected = configs.find((c) => c.id === selectedConfigId)
    if (!selected) {
      setStatus({ kind: 'error', message: 'Selected config not found.' })
      return
    }

    setForm(toPayload(selected))
    setUploadedConfigFileName('')
    const response = await run(() => api.selectConfig(selected.id), 'Config selected in backend runtime.')
    if (!response) return
    setActiveConfig(response.active)
  }

  async function deleteSelectedConfig() {
    if (selectedConfigId === '') {
      setStatus({ kind: 'info', message: 'Select a config to delete first.' })
      return
    }

    const response = await run(
      () => api.deleteConfig(selectedConfigId),
      'Selected configuration deleted.',
    )
    if (!response) return

    if (activeConfig?.id === selectedConfigId) {
      setActiveConfig(null)
    }
    setSelectedConfigId('')
    await refreshConfigs()
  }

  function downloadConfig(format: 'json' | 'yaml') {
    const data = JSON.stringify(form, null, 2)
    const content = format === 'json' ? data : stringifyYaml(JSON.parse(data))
    const mime = format === 'json' ? 'application/json' : 'application/x-yaml'
    const title = (form.database_server.title ?? 'database_config').trim() || 'database_config'
    const safeTitle = title.replace(/[^A-Za-z0-9_-]+/g, '_')

    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${safeTitle}.${format}`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  async function connectWithPayload(payloadConfig: ConfigPayload, createDb: boolean) {
    const payload = {
      ...payloadConfig,
      save: true,
      create_chat_database_if_missing: createDb,
    }

    const response = await run(
      () => api.connect(payload),
      createDb
        ? 'Connected and chat database ensured.'
        : 'Connected successfully.',
    )

    if (!response) return
    setActiveConfig(response.active)
    await refreshConfigs()
  }

  async function connect(createDb: boolean) {
    await connectWithPayload(form, createDb)
  }

  function triggerConfigUpload(connectAfterUpload: boolean) {
    setPendingUploadConnect(connectAfterUpload)
    configUploadRef.current?.click()
  }

  async function handleConfigUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    const uploadedShouldConnect = pendingUploadConnect
    setPendingUploadConnect(false)

    try {
      const text = await file.text()
      const parsed = parseUploadedConfigText(file.name, text)
      const normalized = normalizeUploadedConfig(parsed)
      if (!normalized) {
        setStatus({ kind: 'error', message: 'Unsupported config format in uploaded file.' })
        return
      }

      setForm(normalized)
      setSelectedConfigId('')
      setView('settings')
      setUploadedConfigFileName(file.name)

      if (uploadedShouldConnect) {
        await connectWithPayload(normalized, false)
      } else {
        const saved = await run(
          () => api.saveConfig(normalized),
          'Config uploaded, saved, and shown in input fields.',
        )
        if (!saved) return
        setSelectedConfigId(saved.id)
        await refreshConfigs()
      }
    } catch {
      setStatus({ kind: 'error', message: 'Failed to read uploaded JSON/YAML config.' })
    }
  }

  async function loadExamples() {
    const data = await run(() => api.listExamples(), 'Validator examples loaded.')
    if (!data) return
    setExamples(data.items)
  }

  async function validateExample(questionId: number) {
    const response = await run(() => api.validateExample(questionId), `Validated question ${questionId}.`)
    if (!response) return
    await loadExamples()
  }

  async function removeExample(questionId: number) {
    const response = await run(() => api.deleteExample(questionId), `Deleted question ${questionId}.`)
    if (!response) return
    await loadExamples()
  }

  async function loadExampleIntoChat(example: ExampleRow) {
    // Create a new session for this example
    const newSession = await run(
      () => api.createSession({
        ...sessionForm,
        label: makeAutoSessionLabel(`Validator Example ${example.question_id}`),
        referenced_session_id: null,
      }),
      'New session created.',
    )
    if (!newSession?.session_id) return

    const seeded = await run(
      () => api.saveMessagePair({
        session_id: newSession.session_id,
        user_prompt: example.question_content,
        agent_response: `Loaded validator example ${example.question_id}.\n\n\`\`\`sql\n${example.sql}\n\`\`\``,
        model_name: selectedModel || 'api',
      }),
      'Example question and SQL inserted into chat history.',
    )
    if (!seeded) return

    const historyResponse = await run(
      () => api.getSessionHistory(newSession.session_id),
      'Session history loaded.',
    )

    setSessionId(newSession.session_id)
    setChatQuestion(example.question_content)
    setChatSql(example.sql)
    setHistory(historyResponse?.items ?? [])

    // Update past sessions list
    setPastSessions((prev) => {
      if (prev.some((s) => s.id === newSession.session_id)) return prev
      const label = `Validator Example ${example.question_id}`
      return [{ id: newSession.session_id, label, timestamp: '' }, ...prev]
    })

    setView('chat')
    setStatus({ kind: 'success', message: `Loaded validator example ${example.question_id} into new chat session.` })
  }

  async function loadMetadata() {
    const [tablesResponse, columnsResponse, knowledgeResponse] = await Promise.all([
      run(() => api.listTables()),
      run(() => api.listColumns()),
      run(() => api.listKnowledge()),
    ])

    if (tablesResponse) setTables(tablesResponse.items)
    if (columnsResponse) setColumns(columnsResponse.items)
    if (knowledgeResponse) setKnowledge(knowledgeResponse.items)

    if (tablesResponse || columnsResponse || knowledgeResponse) {
      setStatus({ kind: 'success', message: 'Metadata loaded from backend.' })
    }
  }

  async function loadVectorstoreStats() {
    const response = await run(() => api.vectorstoreStats(), 'Vectorstore stats loaded.')
    if (!response) return
    setVectorStats(response.collections)
  }

  async function searchVectorstore() {
    const query = vectorQuery.trim()
    if (!query) {
      setStatus({ kind: 'info', message: 'Vectorstore query is required.' })
      return
    }

    const collections = vectorCollectionsInput
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    const response = await run(
      () => api.vectorstoreSearch({ query, collections: collections.length > 0 ? collections : undefined, k: Math.max(1, vectorK) }),
      'Vectorstore search completed.',
    )
    if (!response) return
    setVectorResults(response.results)
  }

  async function resyncVectorstore() {
    const response = await run(
      () => api.vectorstoreResync(),
      'Vectorstore resynced from examples, metadata, and knowledge.',
    )
    if (!response) return
    await loadVectorstoreStats()
  }

  async function resetVectorstore() {
    const response = await run(
      () => api.vectorstoreReset(),
      'Vectorstore reset completed.',
    )
    if (!response) return
    setVectorResults({})
    await loadVectorstoreStats()
  }

  function clearKnowledgeForm() {
    setKnowledgeReference('')
    setKnowledgeContent('')
    setEditingKnowledgeId(null)
    setEditingKnowledgeReference('')
  }

  function startEditingKnowledge(item: KnowledgeRow) {
    setEditingKnowledgeId(item.id)
    setEditingKnowledgeReference(item.reference)
    setKnowledgeReference(item.reference)
    setKnowledgeContent(item.content ?? '')
  }

  async function saveKnowledge() {
    const reference = knowledgeReference.trim()
    const content = knowledgeContent.trim()

    if (!reference) {
      setStatus({ kind: 'info', message: 'Knowledge reference is required.' })
      return
    }

    if (!content) {
      setStatus({ kind: 'info', message: 'Knowledge content is required.' })
      return
    }

    const response = editingKnowledgeId
      ? await run(
          () => api.updateKnowledge(editingKnowledgeReference || reference, { reference, content }),
          `Knowledge ${reference} updated.`,
        )
      : await run(
          () => api.createKnowledge({ reference, content }),
          `Knowledge ${reference} created.`,
        )

    if (!response) return
    clearKnowledgeForm()
    await loadMetadata()
  }

  async function removeKnowledge(item: KnowledgeRow) {
    const response = await run(
      () => api.deleteKnowledge(item.id),
      `Knowledge ${item.reference} deleted.`,
    )
    if (!response) return

    if (editingKnowledgeId === item.id) {
      clearKnowledgeForm()
    }
    await loadMetadata()
  }

  async function createSession() {
    const response = await run(
      () => api.createSession({ ...sessionForm, referenced_session_id: null }),
      'Chat session created.',
    )
    if (!response) return
    const newId = response.session_id
    setSessionId(newId)
    setPastSessions((prev: Array<{ id: number; label: string; timestamp: string }>) => {
      if (prev.some((s) => s.id === newId)) return prev
      const label = sessionForm.label.trim() || `Session ${newId}`
      return [{ id: newId, label, timestamp: '' }, ...prev]
    })
    const historyResponse = await run(
      () => api.getSessionHistory(newId),
      'Session history loaded.',
    )
    setHistory(historyResponse?.items ?? [])
  }

  async function startNewSessionLocally() {
    const response = await run(
      () => api.createSession({
        ...sessionForm,
        label: sessionForm.label.trim() || makeAutoSessionLabel('Session'),
        referenced_session_id: null,
      }),
      'New session created.',
    )

    if (!response) {
      const usedSessionIds = new Set<number>(pastSessions.map((s) => s.id))
      if (sessionId !== '') usedSessionIds.add(sessionId)
      setSessionId(createRandomUnusedSessionId(usedSessionIds))
      setStatus({ kind: 'info', message: 'Started a new local session. Create session in backend before saving.' })
      return
    }

    const newId = response.session_id
    setSessionId(newId)
    setPastSessions((prev) => {
      if (prev.some((s) => s.id === newId)) return prev
      const label = sessionForm.label.trim() || `Session ${newId}`
      return [{ id: newId, label, timestamp: '' }, ...prev]
    })

    setHistory([])
    setChatQuestion('')
    setChatSql('select 1 as value')
    setAgentResponse('')
    setCodeRunResults(new Map())
    setQueryResult(null)
    setPendingValidationQuestion('')
    setShowValidationConfirm(false)
    setStatus({ kind: 'info', message: `Started new persisted session #${newId}.` })
  }

  async function loadHistory() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set a session id first.' })
      return
    }
    const response = await run(() => api.getSessionHistory(sessionId), 'Session history loaded.')
    if (!response) return
    setHistory(response.items)
  }

  async function undoLastTurn() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set a session id first.' })
      return
    }

    const response = await run(
      () => api.undoSessionLastTurn(sessionId),
      'Last chat turn removed.',
    )
    if (!response) return

    if (response.deleted === 0) {
      setStatus({ kind: 'info', message: 'Nothing to undo for this session.' })
      return
    }

    const historyResponse = await run(
      () => api.getSessionHistory(sessionId),
      'Session history loaded.',
    )
    setHistory(historyResponse?.items ?? [])
  }

  async function loadModels() {
    const response = await run(() => api.listModels(), 'Models loaded.')
    if (!response) return
    setModels(response.items)
    const hasSelected = response.items.some(
      (m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null),
    )
    if (response.items.length > 0 && (!selectedModel || !hasSelected)) {
      setSelectedModel(response.items[0].model_name)
      setSelectedModelProvider(response.items[0].provider ?? null)
    }
  }

  async function addModel() {
    const modelName = newModelName.trim()
    if (!modelName) {
      setStatus({ kind: 'info', message: 'Model name is required.' })
      return
    }
    const response = await run(
      () => api.addModel({ model_name: modelName, provider: newModelProvider.trim() || null }),
      `Model ${modelName} added.`,
    )
    if (!response) return
    setNewModelName('')
    setNewModelProvider('')
    await loadModels()
  }

  async function deleteModel(modelName: string, provider: string | null) {
    const response = await run(
      () => api.deleteModel(modelName, provider),
      `Model ${modelName} deleted.`,
    )
    if (!response) return
    await loadModels()
  }

  function renderQueryResultPanel() {
    if (!queryResult) return null

    return (
      <article>
        <h3>Query result ({queryResult.count})</h3>
        <div className="result-scroll">
          <table>
            <thead>
              <tr>
                <th>#</th>
                {queryResult.columns.map((column, idx) => (
                  <th key={`col-${idx}`}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {queryResult.rows.map((row, idx) => (
                <tr key={`row-${idx}`}>
                  <td>{idx + 1}</td>
                  {row.map((cell, cellIdx) => (
                    <td key={`cell-${idx}-${cellIdx}`}>
                      <pre>{cell === null || cell === undefined ? '' : String(cell)}</pre>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    )
  }

  async function executeQuery() {
    const sql = chatSql.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required.' })
      return
    }
    await executeSqlText(sql, 'sandbox')
  }

  async function generatePlotCodeFromSql() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set or create a session first.' })
      return
    }

    const currentModel =
      models.find((m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null))
      ?? models.find((m) => m.model_name === selectedModel)
    const sqlOverride = chatSql.trim() || undefined
    const plottingRequest = plottingInstructions.trim() || undefined
    const response = await run(
      () => api.generatePlotCode({
        session_id: sessionId,
        model_name: selectedModel || 'api',
        model_provider: currentModel?.provider ?? null,
        sql_override: sqlOverride,
        plotting_request: plottingRequest,
      }),
      'Plot code generated from SQL.',
    )
    if (!response) return

    setEditableAgentSql({})
    setCodeRunResults(new Map())
    setAgentResponse(response.code)
    if (response.sql_used) {
      setChatSql(response.sql_used)
    }
  }

  async function correctSqlFromLastError() {
    const sql = chatSql.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required.' })
      return
    }
    if (!lastSqlError.trim()) {
      setStatus({ kind: 'info', message: 'No SQL error available to correct.' })
      return
    }

    const currentModel =
      models.find((m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null))
      ?? models.find((m) => m.model_name === selectedModel)
    const response = await run(
      () => api.correctSql({
        sql,
        error_message: lastSqlError,
        history_summary: sqlCorrectionHistorySummary,
        model_name: selectedModel || 'api',
        model_provider: currentModel?.provider ?? null,
      }),
      'SQL correction generated.',
    )
    if (!response) return

    setChatSql(response.corrected_sql)
    setEditableAgentSql({})
    setCodeRunResults(new Map())
    setAgentResponse(response.raw_response)
  }

  async function runCodeChunk(chunkIdx: number, code: string) {
    // Look for the most recent SQL block before this code block
    let sql: string | undefined
    for (let i = chunkIdx - 1; i >= 0; i--) {
      const chunk = parsedAgentResponse[i]
      if (chunk && chunk.type === 'code' && chunk.language === 'sql') {
        sql = chunk.content
        break
      }
    }
    
    const result = await run(() => api.runCode({ code, sql }), '')
    if (!result) return
    setCodeRunResults((prev) => new Map(prev).set(chunkIdx, result))
  }

  async function executeSqlText(sqlText: string, source: string = 'sandbox') {
    const sql = sqlText.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required.' })
      return
    }

    setQueryResultSource(source)
    setQueryResult(null)
    setChatSql(sql)
    setLoading(true)
    setStatus(null)
    setLastSqlError('')
    try {
      const response = await api.runQuery({ sql })
      setQueryResult(response)
      setStatus({ kind: 'success', message: 'Query executed successfully.' })
      if (!agentResponse.trim()) {
        setAgentResponse(`Query returned ${response.count} rows.`)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus({ kind: 'error', message })
      setLastSqlError(message)
    } finally {
      setLoading(false)
    }
  }

  async function streamChatResponse() {
    const prompt = chatQuestion.trim()
    if (!prompt) {
      setStatus({ kind: 'info', message: 'Question is required for streaming.' })
      return
    }

    const currentModel =
      models.find((m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null))
      ?? models.find((m) => m.model_name === selectedModel)

    setLoading(true)
    setStatus(null)
    setEditableAgentSql({})
    setAgentResponse('')

    try {
      const response = await api.streamChat({
        prompt,
        model_name: selectedModel || 'api',
        model_provider: currentModel?.provider ?? null,
        session_id: sessionId === '' ? null : sessionId,
      })

      if (!response.body) {
        throw new Error('Streaming response body is empty.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (!value) continue
        const chunk = decoder.decode(value, { stream: true })
        if (chunk) {
          setAgentResponse((prev) => `${prev}${chunk}`)
        }
      }

      // Auto-refresh history so the latest user/assistant pair appears immediately.
      if (sessionId !== '') {
        try {
          const historyResponse = await api.getSessionHistory(sessionId)
          setHistory(historyResponse.items)
        } catch {
          // Keep stream success state even if history refresh fails.
        }
      }

      setStatus({ kind: 'success', message: 'Streaming response completed.' })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus({ kind: 'error', message })
    } finally {
      setLoading(false)
    }
  }

  async function saveQueryForValidation() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set or create a session first.' })
      return
    }

    const sql = chatSql.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required to save for validation.' })
      return
    }

    const currentModel =
      models.find((m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null))
      ?? models.find((m) => m.model_name === selectedModel)
    const response = await run(
      () => api.saveQueryForValidation({
        session_id: sessionId,
        sql,
        model_name: selectedModel || 'api',
        model_provider: currentModel?.provider ?? null,
        preview_only: true,
      }),
      'Suggested final question generated. Confirm to save.',
    )
    if (!response) return

    setPendingValidationQuestion(response.question_content || '')
    setShowValidationConfirm(true)
  }

  async function confirmSaveQueryForValidation() {
    if (sessionId === '') {
      setStatus({ kind: 'info', message: 'Set or create a session first.' })
      return
    }

    const sql = chatSql.trim()
    if (!sql) {
      setStatus({ kind: 'info', message: 'SQL is required to save for validation.' })
      return
    }

    const finalQuestion = pendingValidationQuestion.trim()
    if (!finalQuestion) {
      setStatus({ kind: 'info', message: 'Suggested final question is empty.' })
      return
    }

    const currentModel =
      models.find((m) => m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null))
      ?? models.find((m) => m.model_name === selectedModel)
    const response = await run(
      () => api.saveQueryForValidation({
        session_id: sessionId,
        sql,
        model_name: selectedModel || 'api',
        model_provider: currentModel?.provider ?? null,
        question_content: finalQuestion,
        preview_only: false,
      }),
      'Query saved for validation.',
    )
    if (!response) return

    setChatQuestion(finalQuestion)
    setShowValidationConfirm(false)
    setPendingValidationQuestion('')

    await loadHistory()
    await loadExamples()
  }

  function cancelValidationSave() {
    setShowValidationConfirm(false)
    setPendingValidationQuestion('')
  }

  function updateDb<K extends keyof ConfigPayload['database_server']>(key: K, value: ConfigPayload['database_server'][K]) {
    setForm((prev) => ({
      ...prev,
      database_server: {
        ...prev.database_server,
        [key]: value,
      },
    }))
  }

  function updateChat<K extends keyof ConfigPayload['chat_database_server']>(key: K, value: ConfigPayload['chat_database_server'][K]) {
    setForm((prev) => ({
      ...prev,
      chat_database_server: {
        ...prev.chat_database_server,
        [key]: value,
      },
    }))
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>KooplexQuery Migration UI</h1>
        <div className="active-badge">Active: {activeSummary}</div>
      </header>

      <nav className="nav">
        <button className={view === 'settings' ? 'active' : ''} onClick={() => setView('settings')}>Settings</button>
        <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>Chat</button>
        <button className={view === 'validator' ? 'active' : ''} onClick={() => setView('validator')}>Validator</button>
        <button className={view === 'metadata' ? 'active' : ''} onClick={() => setView('metadata')}>Metadata</button>
        <button className={view === 'search-knowledge' ? 'active' : ''} onClick={() => setView('search-knowledge')}>Search Knowledge</button>
      </nav>

      {status && (
        <div className={`status ${status.kind}`}>
          {status.message}
        </div>
      )}

      <main className="panel">
        {view === 'settings' && (
          <section>
            <h2>Backend Settings & Connect</h2>
            <div className="row-actions" style={{marginBottom:'16px'}}>
              <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} style={{display:'flex',alignItems:'center',gap:'6px'}}>
                {theme === 'light' ? '🌙 Dark mode' : '☀️ Light mode'}
              </button>
            </div>
            <input
              ref={configUploadRef}
              type="file"
              accept="application/json,text/yaml,application/x-yaml,.json,.yaml,.yml"
              className="hidden-input"
              onChange={handleConfigUpload}
            />
            <div className="row-actions">
              <button disabled={loading} onClick={refreshConfigs}>📂 Load saved configs</button>
              <button disabled={loading} onClick={refreshActiveConfig}>📂 Load active config</button>
              <button disabled={loading} onClick={() => triggerConfigUpload(false)}>Upload config file</button>
              <button disabled={loading} onClick={() => triggerConfigUpload(true)}>Upload + connect</button>
            </div>
            {uploadedConfigFileName && (
              <p className="upload-indicator">Loaded from file: {uploadedConfigFileName}</p>
            )}

            <div className="field-row">
              <label>Saved config</label>
              <select
                value={selectedConfigId}
                onChange={(e) => setSelectedConfigId(e.target.value ? Number(e.target.value) : '')}
              >
                <option value="">Select saved configuration</option>
                {configs.map((cfg) => (
                  <option key={cfg.id} value={cfg.id}>
                    {cfg.db_title ?? 'Untitled'} - {cfg.db_hostname}:{cfg.db_port}/{cfg.db_database}
                  </option>
                ))}
              </select>
              <button disabled={loading || selectedConfigId === ''} onClick={loadFromSelected}>Use selected</button>
              <button disabled={loading || selectedConfigId === ''} className="danger" onClick={deleteSelectedConfig}>🗑 Delete selected</button>
            </div>

            <div className="row-actions">
              <button disabled={loading} onClick={() => downloadConfig('json')}>📥 Download JSON</button>
              <button disabled={loading} onClick={() => downloadConfig('yaml')}>📥 Download YAML</button>
            </div>

            

            <div className="settings-section">
            <h3 className="settings-section-title">Database Configuration</h3>
            <div className="grid two-col">
              <fieldset>
                <legend>Source database</legend>
                <label>Title<input value={form.database_server.title ?? ''} onChange={(e) => updateDb('title', e.target.value)} /></label>
                <label>Host<input value={form.database_server.hostname ?? ''} onChange={(e) => updateDb('hostname', e.target.value)} /></label>
                <label>Port<input type="number" value={form.database_server.port ?? 5432} onChange={(e) => updateDb('port', Number(e.target.value) || 5432)} /></label>
                <label>Database<input value={form.database_server.database ?? ''} onChange={(e) => updateDb('database', e.target.value)} /></label>
                <label>User<input value={form.database_server.user ?? ''} onChange={(e) => updateDb('user', e.target.value)} /></label>
                <label>Password<input type="password" value={form.database_server.password ?? ''} onChange={(e) => updateDb('password', e.target.value)} /></label>
                <label>Schema<input value={form.database_server.schema ?? ''} onChange={(e) => updateDb('schema', e.target.value)} /></label>
                <label>Type<input value={form.database_server.type ?? ''} onChange={(e) => updateDb('type', e.target.value)} /></label>
                <label>URL (optional)<input value={form.database_server.url ?? ''} onChange={(e) => updateDb('url', e.target.value)} /></label>
              </fieldset>

              <fieldset>
                <legend>Chat metadata database</legend>
                <label>Host<input value={form.chat_database_server.hostname ?? ''} onChange={(e) => updateChat('hostname', e.target.value)} /></label>
                <label>Port<input type="number" value={form.chat_database_server.port ?? 5432} onChange={(e) => updateChat('port', Number(e.target.value) || 5432)} /></label>
                <label>Database<input value={form.chat_database_server.database ?? ''} onChange={(e) => updateChat('database', e.target.value)} /></label>
                <label>User<input value={form.chat_database_server.user ?? ''} onChange={(e) => updateChat('user', e.target.value)} /></label>
                <label>Password<input type="password" value={form.chat_database_server.password ?? ''} onChange={(e) => updateChat('password', e.target.value)} /></label>
                <label>Schema<input value={form.chat_database_server.schema ?? ''} onChange={(e) => updateChat('schema', e.target.value)} /></label>
              </fieldset>
            </div>

            <div className="row-actions">
              <button disabled={loading} onClick={() => connect(false)}>Connect</button>
              <button disabled={loading} onClick={() => connect(true)}>Create chat DB + connect</button>
            </div>
            </div>{/* end settings-section db */}

            <div className="settings-section">
            <h3 className="settings-section-title">Model Selection</h3>
            <fieldset>
              <legend>Models</legend>
              <div className="row-actions">
                <button disabled={loading} onClick={loadModels}>Refresh models</button>
              </div>
              <label>New model name
                <input
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  onKeyDown={(e) => onEnterPress(e, () => { void addModel() })}
                />
              </label>
              <label>Provider
                <input
                  value={newModelProvider}
                  onChange={(e) => setNewModelProvider(e.target.value)}
                  onKeyDown={(e) => onEnterPress(e, () => { void addModel() })}
                />
              </label>
              <div className="row-actions">
                <button disabled={loading} onClick={addModel}>Add model</button>
              </div>
              {models.length > 0 && (
                <ul className="list">
                  {models.map((m, idx) => (
                    <li
                      key={`${m.model_name}-${m.provider ?? 'none'}-${idx}`}
                      className={`model-list-item ${m.reachable ? 'model-available' : 'model-unreachable'}`}
                    >
                      <div className="model-list-info">
                        <strong>{m.model_name}</strong>
                        {m.provider && <span>{m.provider}</span>}
                      </div>
                      <button
                        disabled={loading}
                        className="danger icon-btn"
                        title={`Delete ${m.model_name}`}
                        onClick={() => deleteModel(m.model_name, m.provider ?? null)}
                      >🗑</button>
                    </li>
                  ))}
                </ul>
              )}
            </fieldset>
            </div>{/* end settings-section models */}
          </section>
        )}

        {view === 'chat' && (
          <section>
            <h2>Chat</h2>
            <div className="chat-layout">
              <aside className="chat-sidebar">
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => { void startNewSessionLocally() }}
                  style={{ width: '100%', marginBottom: '10px' }}
                >
                  New session
                </button>
                <details className="sidebar-section">

                  <summary className="sidebar-section-summary">Session</summary>
                  <div className="sidebar-section-body">
                  <label>Session ID
                    <input
                      type="number"
                      value={sessionId}
                      onChange={(e) => setSessionId(e.target.value ? Number(e.target.value) : '')}
                      onKeyDown={(e) => onEnterPress(e, () => { void loadHistory() })}
                    />
                  </label>
                  <label>Username
                    <input
                      value={sessionForm.username}
                      onChange={(e) => setSessionForm((prev) => ({ ...prev, username: e.target.value }))}
                    />
                  </label>
                  <label>Email
                    <input
                      value={sessionForm.email}
                      onChange={(e) => setSessionForm((prev) => ({ ...prev, email: e.target.value }))}
                    />
                  </label>
                  <label>Label
                    <input
                      value={sessionForm.label}
                      onChange={(e) => setSessionForm((prev) => ({ ...prev, label: e.target.value }))}
                    />
                  </label>
                  <label>Meta
                    <input
                      value={sessionForm.meta}
                      onChange={(e) => setSessionForm((prev) => ({ ...prev, meta: e.target.value }))}
                    />
                  </label>
                  <div className="row-actions">
                    <button disabled={loading} onClick={createSession}>Create session</button>
                    <button disabled={loading} onClick={loadHistory}>📂 Load history</button>
                  </div>
                  </div>
                </details>

                <fieldset>
                  <legend>Model</legend>
                  {models.length === 0 ? (
                    <span className="no-models-hint">No models loaded</span>
                  ) : (
                    <ul className="list model-selector-list">
                      {models.map((m, idx) => (
                        <li
                          key={`${m.model_name}-${m.provider ?? 'none'}-${idx}`}
                          className={`model-select-item ${m.reachable ? 'model-available' : 'model-unreachable'}${m.model_name === selectedModel && (m.provider ?? null) === (selectedModelProvider ?? null) ? ' selected' : ''}`}
                          onClick={() => {
                            setSelectedModel(m.model_name)
                            setSelectedModelProvider(m.provider ?? null)
                          }}
                        >
                          <strong>{m.model_name}</strong>
                          {m.provider && <span>{m.provider}</span>}
                        </li>
                      ))}
                    </ul>
                  )}
                </fieldset>

                <details className="sidebar-section" open>
                  <summary className="sidebar-section-summary">Previous sessions</summary>
                  <div className="sidebar-section-body">
                    <div className="row-actions" style={{marginBottom:'6px'}}>
                      <button disabled={loading} onClick={async () => {
                        const r = await run(() => api.listSessions(), 'Sessions refreshed.')
                        if (r) setPastSessions(r.items.map((s) => ({
                          id: s.session_id,
                          label: s.content ? s.content.slice(0, 60).replace(/\s+/g, ' ').trim() : `Session ${s.session_id}`,
                          timestamp: s.timestamp ?? '',
                        })))
                      }}>↺ Refresh</button>
                    </div>
                    {pastSessions.length === 0 ? (
                      <p className="muted-hint">No sessions yet.</p>
                    ) : (
                      <ul className="session-list">
                        {pastSessions.map((s) => (
                          <li
                            key={s.id}
                            className={`session-list-item${s.id === sessionId ? ' active-session' : ''}`}
                            onClick={async () => {
                              setSessionId(s.id)
                              const r = await run(() => api.getSessionHistory(s.id), 'Session history loaded.')
                              setHistory(r?.items ?? [])
                            }}
                          >
                            <span className="session-list-label">{s.label}</span>
                            <span className="session-list-meta">#{s.id}{s.timestamp ? ` · ${new Date(s.timestamp).toLocaleDateString()}` : ''}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </details>
              </aside>

              <div className="chat-main">
              <article>
                <h3>Chat history ({conversations.length} conversations)</h3>
                <div className="conversation-list">
                  {conversations.map((conv, idx, arr) => {
                    const isLatest = idx === arr.length - 1 && idx !== 0
                    const responseText = conv.assistantMessage?.content ?? ''
                    const segments = splitResponseIntoSegments(responseText)

                    return (
                      <details key={conv.id} className="conversation-item" open={isLatest}>
                        <summary className="conversation-summary">
                          <span className="conversation-label">User:</span>
                          <span className="conversation-title">{shortPreview(conv.firstUserMessage || '(empty message)')}</span>
                        </summary>
                        <div className="conversation-body">
                          {conv.userMessage && (
                            <div className="chat-bubble user">
                              <div className="chat-meta">{conv.userMessage.timestamp}</div>
                              <pre>{conv.userMessage.content}</pre>
                            </div>
                          )}

                          {conv.assistantMessage ? (
                            <div className="chat-bubble assistant">
                              <div className="chat-meta">{conv.assistantMessage.timestamp}</div>
                              {segments.map((segment, segmentIndex) =>
                                segment.kind === 'text' ? (
                                  <pre key={`${conv.id}-text-${segmentIndex}`}>{segment.content}</pre>
                                ) : (
                                  <div key={`${conv.id}-code-${segmentIndex}`} className="code-segment">
                                    <div className="code-segment-header">
                                      <span>{segment.language.toUpperCase()}</span>
                                    </div>
                                    <pre>{segment.content}</pre>
                                    {segment.language === 'sql' && (
                                      <div className="row-actions conversation-actions">
                                        <button
                                          type="button"
                                          disabled={loading}
                                          onClick={() => setChatSql(segment.content)}
                                        >
                                          Use SQL in editor
                                        </button>
                                        <button
                                          type="button"
                                          disabled={loading}
                                          onClick={() => { void executeSqlText(segment.content, `history-${conv.id}-${segmentIndex}`) }}
                                        >
                                          Run SQL
                                        </button>
                                      </div>
                                    )}
                                    {segment.language === 'sql' && queryResultSource === `history-${conv.id}-${segmentIndex}` && renderQueryResultPanel()}
                                  </div>
                                ),
                              )}
                              <div className="row-actions conversation-actions">
                                <button
                                  type="button"
                                  disabled={loading || sessionId === ''}
                                  onClick={() => { void undoLastTurn() }}
                                >
                                  Undo last turn
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div className="chat-bubble assistant pending">
                              <pre>(No assistant response saved yet)</pre>
                            </div>
                          )}
                        </div>
                      </details>
                    )
                  })}
                </div>
              </article>

                <fieldset className="sql-sandbox">
                  <legend>Query + message workflow</legend>
              <label>SQL
                <textarea
                  className="code-area"
                  value={chatSql}
                  onChange={(e) => setChatSql(e.target.value)}
                />
              </label>
              <div className="row-actions">
                <button disabled={loading} onClick={executeQuery}>▶ Run SQL</button>
              </div>
              {queryResultSource === 'sandbox' && renderQueryResultPanel()}
              <label>Plotting instructions (optional)
                <textarea
                  className="code-area"
                  placeholder="Specify any desired chart type, styling, or visualization preferences"
                  value={plottingInstructions}
                  onChange={(e) => setPlottingInstructions(e.target.value)}
                />
              </label>
              <div className="row-actions">
                <button disabled={loading || sessionId === ''} onClick={() => { void generatePlotCodeFromSql() }}>
                  Generate plot code
                </button>
              </div>
              {lastSqlError && (
                <div className="sql-error-panel">
                  <div className="sql-error-header">
                    <strong>SQL execution error</strong>
                    {hasSqlAlchemyError && (
                      <button disabled={loading} onClick={() => { void correctSqlFromLastError() }}>
                        Correct SQL
                      </button>
                    )}
                  </div>
                  <pre>{lastSqlError}</pre>
                </div>
              )}
              {/* <div className="grid two-col">
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={chatPublic}
                    onChange={(e) => setChatPublic(e.target.checked)}
                  />
                  Save query as public
                </label>
              </div> */}
              <div className="agent-response-panel">
                <label>Agent response</label>
                {parsedAgentResponse.length > 0 && (
                  <div className="stream-renderer">
                    {parsedAgentResponse.map((chunk, idx) => {
                      if (chunk.type === 'text') {
                        return (
                          <article className="stream-chunk stream-chunk-text" key={`stream-text-${idx}`}>
                            <pre>{chunk.content}</pre>
                          </article>
                        )
                      }

                      const isSql = chunk.language === 'sql'
                      const isRunnable = !isSql && ['python', 'py', ''].includes(chunk.language)
                      const currentSqlChunk = editableAgentSql[idx] ?? chunk.content
                      const codeResult = codeRunResults.get(idx)
                      return (
                        <article className="stream-chunk stream-chunk-code" key={`stream-code-${idx}`}>
                          <div className="stream-chunk-header">
                            <strong>{isSql ? 'SQL block' : `Code block (${chunk.language})`}</strong>
                          </div>
                          {isSql ? (
                            <textarea
                              className="code-area agent-sql-editor"
                              value={currentSqlChunk}
                              onChange={(e) => {
                                const value = e.target.value
                                setEditableAgentSql((prev) => ({ ...prev, [idx]: value }))
                              }}
                            />
                          ) : (
                            <pre>{chunk.content}</pre>
                          )}
                          {isSql && (
                            <div className="row-actions">
                              <button
                                disabled={loading}
                                onClick={() => setChatSql(currentSqlChunk)}
                              >
                                Use SQL in editor
                              </button>
                              <button
                                disabled={loading}
                                onClick={() => executeSqlText(currentSqlChunk, `agent-${idx}`)}
                              >
                                ▶ Run this SQL
                              </button>
                            </div>
                          )}
                          {isSql && queryResultSource === `agent-${idx}` && renderQueryResultPanel()}
                          {isRunnable && (
                            <div className="row-actions">
                              <button
                                disabled={loading}
                                onClick={() => { void runCodeChunk(idx, chunk.content) }}
                              >
                                ▶ Run code
                              </button>
                            </div>
                          )}
                          {codeResult && (
                            <div className="code-run-output">
                              {codeResult.error && (
                                <pre className="code-run-error">{codeResult.error}</pre>
                              )}
                              {codeResult.stdout && (
                                <pre className="code-run-stdout">{codeResult.stdout}</pre>
                              )}
                              {codeResult.stderr && (
                                <pre className="code-run-stderr">{codeResult.stderr}</pre>
                              )}
                              {codeResult.plot_html && (
                                <iframe
                                  className="code-run-plot"
                                  srcDoc={codeResult.plot_html}
                                  sandbox="allow-scripts"
                                  title={`plot-${idx}`}
                                />
                              )}
                            </div>
                          )}
                        </article>
                      )
                    })}
                  </div>
                )}
              </div>
              <div className="row-actions">
                <button disabled={loading} onClick={saveQueryForValidation}>💾 Save query for validation</button>
                <button
                  type="button"
                  disabled={loading || sessionId === ''}
                  onClick={() => { void undoLastTurn() }}
                >
                  Undo last turn
                </button>
              </div>
                {showValidationConfirm && (
                  <fieldset>
                    <legend>Confirm final question</legend>
                    <p>The LLM suggested this final question (possibly a refined version of the original):</p>
                    <textarea
                      className="code-area"
                      value={pendingValidationQuestion}
                      onChange={(e) => setPendingValidationQuestion(e.target.value)}
                    />
                    <div className="row-actions">
                      <button disabled={loading} onClick={confirmSaveQueryForValidation}>💾 Confirm save</button>
                      <button disabled={loading} onClick={saveQueryForValidation}>Regenerate suggestion</button>
                      <button disabled={loading} className="danger" onClick={cancelValidationSave}>Cancel</button>
                    </div>
                  </fieldset>
                )}
              </fieldset>

              <div className="chat-composer">
                <div className="chat-composer-row">
                  <label className="chat-question-field">Question
                    <input
                      value={chatQuestion}
                      onChange={(e) => setChatQuestion(e.target.value)}
                      placeholder="Natural language question"
                      onKeyDown={(e) => onEnterPress(e, () => { void streamChatResponse() })}
                    />
                  </label>
                  <button disabled={loading} onClick={streamChatResponse}>📤 Submit</button>
                </div>
              </div>
            </div>
          </div>
          </section>
        )}

        {view === 'validator' && (
          <section>
            <h2>Validator</h2>
            <div className="row-actions">
              <button disabled={loading} onClick={loadExamples}>📂 Load examples</button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Question</th>
                  <th>SQL Query</th>
                  <th>Score</th>
                  <th>Public</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {examples
                  .slice()
                  .sort((a, b) => Number(Boolean(a.public)) - Number(Boolean(b.public)))
                  .map((row) => (
                  <tr key={row.question_id}>
                    <td>{row.question_id}</td>
                    <td>{row.type}</td>
                    <td title={row.question_content}>{row.question_content}</td>
                    <td title={row.sql}><code>{row.sql.substring(0, 50)}{row.sql.length > 50 ? '...' : ''}</code></td>
                    <td>{row.score ?? '-'}</td>
                    <td>{String(row.public)}</td>
                    <td className="cell-actions">
                      <button disabled={loading} onClick={() => loadExampleIntoChat(row)}>Load into chat</button>
                      <button disabled={loading} onClick={() => validateExample(row.question_id)}>Validate</button>
                      <button disabled={loading} className="danger" onClick={() => removeExample(row.question_id)}>🗑 Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {view === 'metadata' && (
          <section>
            <h2>Metadata</h2>
            <div className="row-actions">
              <button disabled={loading} onClick={loadMetadata}>📂 Load metadata</button>
            </div>

            <fieldset>
              <legend>Knowledge Manager</legend>
              <label>Reference
                <input
                  value={knowledgeReference}
                  onChange={(e) => setKnowledgeReference(e.target.value)}
                  placeholder="schema | instruction | business_rules"
                />
              </label>
              <label>Content
                <textarea
                  className="code-area"
                  value={knowledgeContent}
                  onChange={(e) => setKnowledgeContent(e.target.value)}
                  placeholder="Paste schema, instructions, or notes"
                />
              </label>
              <div className="row-actions">
                <button disabled={loading} onClick={saveKnowledge}>
                  {editingKnowledgeId ? '💾 Update knowledge' : '💾 Add knowledge'}
                </button>
                <button disabled={loading} onClick={clearKnowledgeForm}>Clear</button>
              </div>
            </fieldset>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(200px, auto) minmax(200px, auto) 2fr', gap: '24px' }}>
              <article>
                <h3>Tables ({tables.length})</h3>
                <ul className="list">
                  {tables.map((t, idx) => (
                    <li key={`${t.table}-${idx}`}>
                      <strong>{t.table}</strong>
                      <span>{t.description ?? '(no description)'}</span>
                    </li>
                  ))}
                </ul>
              </article>

              <article>
                <h3>Columns ({columns.length})</h3>
                <ul className="list">
                  {columns.map((c, idx) => (
                    <li key={`${c.table}-${c.column}-${idx}`}>
                      <strong>{c.table}.{c.column}</strong>
                      <span>{c.type}</span>
                    </li>
                  ))}
                </ul>
              </article>

              <article>
                <h3>Knowledge ({knowledge.length})</h3>
                <ul className="list">
                  {knowledge.map((k) => (
                    <li key={k.id}>
                      <strong>{k.reference}</strong>
                      <div style={{ fontSize: '0.9em', lineHeight: '1.5' }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {k.content?.trim() || '(empty)'}
                        </ReactMarkdown>
                      </div>
                      <div className="row-actions">
                        <button disabled={loading} onClick={() => startEditingKnowledge(k)}>✏️ Edit</button>
                        <button disabled={loading} className="danger" onClick={() => removeKnowledge(k)}>🗑 Delete</button>
                      </div>
                    </li>
                  ))}
                </ul>
              </article>
            </div>
          </section>
        )}

        {view === 'search-knowledge' && (
          <section>
            <h2>Search Knowledge</h2>
            <div className="row-actions">
              <button disabled={loading} onClick={loadVectorstoreStats}>📂 Load vectorstore stats</button>
            </div>

            <fieldset>
              <legend>Vectorstore</legend>
              <label>Search query
                <input
                  value={vectorQuery}
                  onChange={(e) => setVectorQuery(e.target.value)}
                  placeholder="Find similar examples or schema hints"
                  onKeyDown={(e) => onEnterPress(e, () => { void searchVectorstore() })}
                />
              </label>
              <div className="grid two-col">
                <label>Collections (comma-separated, optional)
                  <input
                    value={vectorCollectionsInput}
                    onChange={(e) => setVectorCollectionsInput(e.target.value)}
                    placeholder="examples, docs"
                    onKeyDown={(e) => onEnterPress(e, () => { void searchVectorstore() })}
                  />
                </label>
                <label>Top K
                  <input
                    type="number"
                    value={vectorK}
                    onChange={(e) => setVectorK(Number(e.target.value) || 1)}
                    onKeyDown={(e) => onEnterPress(e, () => { void searchVectorstore() })}
                  />
                </label>
              </div>
              <div className="row-actions">
                <button disabled={loading} onClick={searchVectorstore}>Search vectorstore</button>
                <button disabled={loading} onClick={resyncVectorstore}>Resync vectorstore</button>
                <button disabled={loading} className="danger" onClick={resetVectorstore}>Reset vectorstore</button>
              </div>
              <pre>{JSON.stringify(vectorStats, null, 2)}</pre>
            </fieldset>

            <article>
              <h3>Vectorstore Search Results</h3>
              <pre>{JSON.stringify(vectorResults, null, 2)}</pre>
            </article>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
