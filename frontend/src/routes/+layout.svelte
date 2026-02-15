<script lang="ts">
	import { MettleHeader, MettleFooter } from '$lib/components';
	import '../app.css';

	let { children } = $props();
</script>

<MettleHeader />
{@render children()}
<MettleFooter />
