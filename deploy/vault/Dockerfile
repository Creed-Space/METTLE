FROM hashicorp/vault:1.20@sha256:268bb80aa9c6d13d65fcfa05c0c268caca068952240a8087291a6ce0b66e3a10

EXPOSE 8200
ENV SKIP_SETCAP=true \
    SKIP_CHOWN=true
USER vault
CMD ["server", "-config=/etc/secrets/vault.hcl"]
