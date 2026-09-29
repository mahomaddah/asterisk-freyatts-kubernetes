{{- define "fv.image" -}}
{{ .root.Values.image.registry }}/{{ .name }}:{{ index .root.Values.image.tags .name }}
{{- end -}}

{{- define "fv.labels" -}}
app.kubernetes.io/part-of: freya-voice
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "fv.podSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  seccompProfile: {type: RuntimeDefault}
{{- end -}}

{{- define "fv.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities: {drop: [ALL]}
{{- end -}}
